import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
import subprocess
from tkinterdnd2 import TkinterDnD, DND_FILES
from extract_rallies import TennisRallyProcessor, generate_rally_name, get_video_date

# Set theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class RallyExtractorGUI(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self) # Initialize DnD

        # Window Config
        self.title("Tennis Rally Extractor")
        self.geometry("700x600")
        
        # Variables
        self.video_files = []
        self.output_dir = os.getcwd() # Default to CWD for speed, avoids network scan
        self.is_processing = False
        self.stop_event = threading.Event()
        
        # Enable Drag and Drop for the window
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop_files)
        
        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Content
        self.grid_rowconfigure(2, weight=0) # Actions
        self.grid_rowconfigure(3, weight=0) # Status
        
        self.create_widgets()
        
    def create_widgets(self):
        # --- Header ---
        self.header_frame = ctk.CTkFrame(self, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        self.header_label = ctk.CTkLabel(self.header_frame, text="Tennis Rally Extractor", font=ctk.CTkFont(size=20, weight="bold"))
        self.header_label.pack(pady=15)

        # --- Main Content ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1) # File list expands
        
        # 1. Output Selection
        self.out_frame = ctk.CTkFrame(self.main_frame)
        self.out_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        self.lbl_output = ctk.CTkLabel(self.out_frame, text="Output Folder:", font=ctk.CTkFont(weight="bold"))
        self.lbl_output.pack(side="left", padx=10)
        
        self.entry_output = ctk.CTkEntry(self.out_frame, placeholder_text=self.output_dir)
        self.entry_output.pack(side="left", fill="x", expand=True, padx=10)
        self.entry_output.insert(0, self.output_dir)
        self.entry_output.configure(state="disabled") 
        
        self.btn_select_output = ctk.CTkButton(self.out_frame, text="Browse", width=80, command=self.select_output)
        self.btn_select_output.pack(side="right", padx=10, pady=10)

        # 2. File Selection Header
        self.file_header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.file_header_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 0))
        
        self.lbl_files = ctk.CTkLabel(self.file_header_frame, text="Input Videos (Drag & Drop Supported)", font=ctk.CTkFont(weight="bold"))
        self.lbl_files.pack(side="left")
        
        self.btn_add_files = ctk.CTkButton(self.file_header_frame, text="+ Add Videos", width=100, command=self.select_files)
        self.btn_add_files.pack(side="right")
        
        self.btn_clear_files = ctk.CTkButton(self.file_header_frame, text="Clear", width=60, fg_color="gray", hover_color="darkgray", command=self.clear_files)
        self.btn_clear_files.pack(side="right", padx=10)

        # 3. File List (Scrollable)
        self.scroll_files = ctk.CTkScrollableFrame(self.main_frame, label_text="Files")
        self.scroll_files.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        # --- Actions Area ---
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        
        # Progress
        self.progress_bar = ctk.CTkProgressBar(self.action_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=(15, 5))
        self.progress_bar.set(0)
        
        self.lbl_progress_text = ctk.CTkLabel(self.action_frame, text="Ready to start.", text_color="gray")
        self.lbl_progress_text.pack(pady=(0, 10))
        
        # Build Button Frame
        self.btn_frame = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        self.btn_process = ctk.CTkButton(self.btn_frame, text="START PROCESSING", height=40, font=ctk.CTkFont(size=14, weight="bold"), command=self.start_processing_thread)
        self.btn_process.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_stop = ctk.CTkButton(self.btn_frame, text="PANNIC STOP", height=40, font=ctk.CTkFont(size=14, weight="bold"), fg_color="#D32F2F", hover_color="#B71C1C", command=self.stop_processing, state="disabled")
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # --- Status Footer ---
        self.lbl_status = ctk.CTkLabel(self, text="v1.2 (DnD) - Status: Idle", anchor="w", padx=10, font=ctk.CTkFont(size=10))
        self.lbl_status.grid(row=3, column=0, sticky="ew")

    # --- Logic ---

    def drop_files(self, event):
        data = event.data
        if not data: return
        
        # TkinterDnD sometimes wraps paths in {} if they contain spaces
        # Split logic depends on OS.
        # Simple parser:
        if "{" in data:
            # complex parsing
            import re
            paths = re.findall(r'\{.*?\}|\S+', data)
            cleaned = [p.strip("{}") for p in paths]
        else:
            cleaned = data.split()
            
        new_files = [f for f in cleaned if os.path.isfile(f) and f not in self.video_files]
        if new_files:
            self.video_files.extend(new_files)
            self.update_file_list()

    def select_files(self):
        # Force initial directory to Movies or Home to avoid scanning network drives which lags
        initial = os.path.expanduser("~/Movies")
        if not os.path.exists(initial):
            initial = os.path.expanduser("~")
            
        files = filedialog.askopenfilenames(
            title="Select Video Files", 
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")],
            initialdir=initial
        )
        if files:
            # Avoid duplicates
            current_files = set(self.video_files)
            new_files = [f for f in files if f not in current_files]
            self.video_files.extend(new_files)
            self.update_file_list()

    def clear_files(self):
        self.video_files = []
        self.update_file_list()
        
    def update_file_list(self):
        # Clear scrollable frame widgets
        for widget in self.scroll_files.winfo_children():
            widget.destroy()
            
        for i, f in enumerate(self.video_files):
            # Row layout
            row = ctk.CTkFrame(self.scroll_files, fg_color="transparent")
            row.pack(fill="x", pady=2)
            
            lbl = ctk.CTkLabel(row, text=f"{i+1}. {os.path.basename(f)}", anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            
        self.lbl_files.configure(text=f"Input Videos ({len(self.video_files)} selected)")
        self.save_default_output()


    def select_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            self.entry_output.configure(state="normal")
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, self.output_dir)
            self.entry_output.configure(state="disabled")
            
    def save_default_output(self):
        # If no output selected, use parent of first video
        if self.output_dir == os.getcwd() and self.video_files:
            parent = os.path.dirname(self.video_files[0])
            self.output_dir = parent
            self.entry_output.configure(state="normal")
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, self.output_dir)
            self.entry_output.configure(state="disabled")

    def log(self, message, progress=None):
        def _update():
            self.lbl_status.configure(text=f"Status: {message}")
            self.lbl_progress_text.configure(text=message)
            if progress is not None:
                self.progress_bar.set(progress)
        self.after(0, _update)

    def start_processing_thread(self):
        if not self.video_files:
            messagebox.showwarning("No Files", "Please select at least one video file.")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.stop_event.clear()
        
        # UI State
        self.btn_process.configure(state="disabled", text="Processing...")
        self.btn_add_files.configure(state="disabled")
        self.btn_clear_files.configure(state="disabled")
        self.btn_select_output.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress_bar.set(0)
        
        thread = threading.Thread(target=self.run_processing)
        thread.start()

    def stop_processing(self):
        if self.is_processing:
            if messagebox.askyesno("Stop", "Are you sure you want to stop processing?"):
                self.stop_event.set()
                self.log("Stopping... handling cleanup...", 0)
                self.btn_stop.configure(state="disabled")

    def run_processing(self):
        try:
            all_clip_paths = []
            
            # 1. Detect Format (Standardization)
            target_format = None
            if self.video_files:
                first_vid = self.video_files[0]
                try:
                    import cv2
                    cap = cv2.VideoCapture(first_vid)
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        target_format = {'width': w, 'height': h, 'fps': fps}
                        self.log(f"Targeting: {w}x{h} @ {fps:.1f}fps", 0)
                    cap.release()
                except Exception as e:
                    print(f"Format detection failed: {e}")

            # 2. Process Loop
            total_videos = len(self.video_files)
            
            for i, video_path in enumerate(self.video_files):
                if self.stop_event.is_set():
                    break
                    
                video_name = os.path.basename(video_path)
                self.log(f"Processing {i+1}/{total_videos}: {video_name}...")
                
                # Progress Adapter
                def video_progress_adapter(fraction, msg):
                    global_frac = (i + fraction) / total_videos
                    self.log(f"[{i+1}/{total_videos}] {video_name}: {msg}", global_frac)
                
                processor = TennisRallyProcessor(
                    video_path, 
                    self.output_dir, 
                    keep_clips=True,
                    progress_callback=video_progress_adapter,
                    stop_event=self.stop_event,
                    target_format=target_format
                ) 
                
                clips = processor.process_video()
                
                if self.stop_event.is_set():
                    break
                    
                all_clip_paths.extend(clips)
                
            # 3. Finalize
            if self.stop_event.is_set():
                self.log("Stopped by user.", 0)
                self.after(0, lambda: messagebox.showinfo("Stopped", "Processing stopped by user."))
                return

            if all_clip_paths:
                self.log(f"Merging {len(all_clip_paths)} clips...", 1.0)
                
                first_video = self.video_files[0]
                date_str = get_video_date(first_video)
                output_name = f"{date_str}_Merged_Rallies.mp4"
                
                processor = TennisRallyProcessor(first_video, self.output_dir)
                merged_path = processor.merge_clips(all_clip_paths, output_name)
                
                if merged_path:
                    self.log("Transferring metadata...", 1.0)
                    processor.transfer_metadata(first_video, merged_path)
                    
                    self.log("Done!", 1.0)
                    msg = f"Processing Complete!\n\nSaved to:\n{merged_path}"
                    self.after(0, lambda: messagebox.showinfo("Success", msg))
                    subprocess.call(["open", "-R", merged_path])
                else:
                    self.log("Error during merge.", 0)
                    self.after(0, lambda: messagebox.showerror("Error", "Failed to merge clips."))
            else:
                self.log("No rallies found.", 0)
                self.after(0, lambda: messagebox.showinfo("Info", "No rallies met the criteria."))
                
        except Exception as e:
            self.log(f"Error: {str(e)}", 0)
            self.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{str(e)}"))
            import traceback
            traceback.print_exc()
            
        finally:
            self.is_processing = False
            self.after(0, self.reset_ui)

    def reset_ui(self):
        self.btn_process.configure(state="normal", text="START PROCESSING")
        self.btn_add_files.configure(state="normal")
        self.btn_clear_files.configure(state="normal")
        self.btn_select_output.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if not self.is_processing:
             pass

if __name__ == "__main__":
    app = RallyExtractorGUI()
    app.mainloop()
