import csv

print("# Timeline and Movement Energy Analysis")
print("This breaks down the video into 5-second chunks to observe metrics.")
print("")
print("| Time (s) | Avg Player Height | Max Arm Energy | Avg Hip Vel | Max Action Score | Active Rows |")
print("|---|---|---|---|---|---|")

try:
    with open('test_diagnostic/debug_metrics.csv', 'r') as f:
        reader = csv.DictReader(f)
        chunks = {}
        for row in reader:
            t = float(row['Time(s)'])
            t_int = int(t // 5) * 5
            if t_int not in chunks:
                chunks[t_int] = []
            chunks[t_int].append(row)
            
    for t_int in sorted(chunks.keys()):
        rows = chunks[t_int]
        heights = [float(r['Player_Height']) for r in rows if r['Player_Height']]
        arms = [float(r['Smoothed_Arm']) for r in rows if r['Smoothed_Arm']]
        hips = [float(r['Hip_Vel']) for r in rows if r['Hip_Vel']]
        actions = [float(r['Action_Score']) for r in rows if r['Action_Score']]
        
        avg_h = sum(heights)/len(heights) if heights else 0
        max_a = max(arms) if arms else 0
        avg_hip = sum(hips)/len(hips) if hips else 0
        max_act = max(actions) if actions else 0
        
        print(f"| {t_int} - {t_int+5} | {avg_h:.2f} | {max_a:.2f} | {avg_hip:.2f} | {max_act:.2f} | {len(rows)} |")

except Exception as e:
    print(f"Error reading csv: {e}")
