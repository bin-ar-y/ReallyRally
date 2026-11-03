// Read-only motion input, separate editable reviewer evidence. Runs offline.
const SHOT_EVENTS = {
    serve: {loading:"Loading starts", foot:"Setup front foot", release:"Ball release", apex:"Toss apex", contact:"Ball at contact"},
    forehand: {ready:"Ready", prepared:"Preparation ends", contact:"Ball at contact", tip1:"Earlier racket tip", tip2:"Later racket tip"}
};
for(const events of Object.values(SHOT_EVENTS)){
    events.early="Earliest plausible impact (optional)";
    events.late="Latest plausible impact (optional)";
}
function contactProblem(data, marks){
    if(!marks.early && !marks.late)return null;
    if(!marks.early || !marks.late || marks.early.frame>data.impact || marks.late.frame<data.impact)
        return "Bracket the selected impact with both the earliest and latest plausible frames.";
    return marks.early.frame===marks.late.frame?null:"Contact is uncertain (frames "+marks.early.frame+"–"+marks.late.frame+").";
}
const POINT_EVENTS = new Set(["foot","release","apex","contact","tip1","tip2"]);
const shotKind = stroke => String(stroke).trim().toLowerCase();
const median = xs => {
    const sorted = [...xs].sort((a,b)=>a-b), mid=Math.floor(sorted.length/2);
    return sorted.length%2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
};

function phaseSeries(data, key, first, last) {
    const i=data.frames.indexOf(first), j=data.frames.indexOf(last);
    if(i<0 || j<=i)return {reason:"Mark the phase start before its end."};
    const samples=[];
    const nominal=median(data.times.slice(1).map((t,k)=>t-data.times[k]));
    for(let n=i;n<=j;n++){
        const value=data.kinematics[n][key];
        if(!Number.isFinite(value))return {reason:"Missing or inconsistent landmarks at frame "+data.frames[n]+".", frames:[data.frames[n]]};
        if(n>i && (data.frames[n]!==data.frames[n-1]+1 || data.times[n]-data.times[n-1]>2.5*nominal ||
            data.times[n]-data.times[n-1]<.25*nominal))return {reason:"Timing gap at frame "+data.frames[n]+".",frames:[data.frames[n]]};
        samples.push({frame:data.frames[n],t:data.times[n],v:value});
    }
    return {samples};
}

function kneeExcursion(data, side, first, last) {
    const series=phaseSeries(data,side+"_knee_flexion_deg",first,last);
    if(!series.samples)return series;
    const points=series.samples;
    const deepest=points.reduce((max,p)=>p.v>max.v?p:max);
    if(deepest===points[0] || deepest===points.at(-1))
        return {reason:"Deepest bend is on the phase boundary. Check the loading mark.", frames:[deepest.frame]};
    const extended=points.slice(points.indexOf(deepest)+1).reduce((min,p)=>p.v<min.v?p:min);
    return {degrees:deepest.v-extended.v, flexion_deg:deepest.v, extension_deg:extended.v,
            ms_before_contact:1000*(points.at(-1).t-extended.t),frames:[deepest.frame,extended.frame,last]};
}

function turnSeries(data,key,a,b,sign){
    const series=phaseSeries(data,key,a,b);
    if(!series.samples)return series;
    let previous=series.samples[0].v, sum=0;
    for(const p of series.samples){
        const change=((p.v-previous+540)%360)-180;
        previous=p.v;sum+=sign*change;p.v=sum;
    }
    return series;
}

// Local least-squares slopes use elapsed time, not a presumed frame rate.
function peakTurn(samples,radius){
    const rates=[];
    for(let i=radius;i<samples.length-radius;i++){
        const window=samples.slice(i-radius,i+radius+1);
        const mean=window.reduce((s,p)=>s+p.t,0)/window.length;
        const denom=window.reduce((s,p)=>s+(p.t-mean)**2,0);
        if(denom<=0)return null;
        rates.push({p:samples[i],rate:window.reduce((s,p)=>s+(p.t-mean)*p.v,0)/denom});
    }
    if(rates.length<3)return null;
    const peak=rates.reduce((a,b)=>b.rate>a.rate?b:a);
    if(peak.rate<=0 || peak===rates[0] || peak===rates.at(-1))return null;
    // Comparable peaks far apart cannot be distinguished reliably.
    if(rates.some(r=>Math.abs(r.p.frame-peak.p.frame)>2 && r.rate>=.98*peak.rate))return null;
    return peak.p;
}

function rotationReview(data,marks,context){
    if(!context.level)return {reason:"Confirm a fixed, level camera before interpreting headings about vertical."};
    if(!["left","right"].includes(context.hand))return {reason:"Select the hitting hand."};
    if(!marks.ready || !marks.prepared)return {reason:"Mark ready and end of preparation."};
    if(!(marks.ready.frame<marks.prepared.frame && marks.prepared.frame<data.impact))
        return {reason:"Use ready → end of preparation → impact order."};
    const sign=context.hand==="right"?1:-1;
    const report={};
    for(const name of ["pelvis","shoulder"]){
        const key=name+"_heading_deg";
        const prep=turnSeries(data,key,marks.ready.frame,marks.prepared.frame,sign);
        report[name+"_turn"]=prep.samples ? {degrees:prep.samples.at(-1).v,frames:[marks.ready.frame,marks.prepared.frame]} : prep;
        const uncertain=contactProblem(data,marks);
        if(uncertain){report[name+"_peak"]={reason:uncertain};continue;}
        const swing=turnSeries(data,key,marks.prepared.frame,data.impact,sign);
        if(!swing.samples){report[name+"_peak"]=swing;continue;}
        const p=peakTurn(swing.samples,1), q=peakTurn(swing.samples,2);
        // Perturb the end by one frame as well as varying the derivative window.
        const short=peakTurn(swing.samples.slice(0,-1),1);
        const extended=turnSeries(data,key,marks.prepared.frame,data.impact+1,sign);
        const long=extended.samples && peakTurn(extended.samples,1);
        if(!p || !q || !short || !long || Math.max(p.frame,q.frame,short.frame,long.frame)-Math.min(p.frame,q.frame,short.frame,long.frame)>1){
            report[name+"_peak"]={reason:"Forward peak unresolved at this sampling rate or unstable across windows."};
        }else{
            const contact=swing.samples.at(-1);
            report[name+"_peak"]={ms_before_contact:1000*(contact.t-p.t),frames:[p.frame],
                uncertainty:[Math.min(p.frame,q.frame,short.frame,long.frame)-1,Math.max(p.frame,q.frame,short.frame,long.frame)+1]};
        }
    }
    const pelvis=report.pelvis_peak, shoulder=report.shoulder_peak;
    if(pelvis.uncertainty && shoulder.uncertainty){
        report.interval_ms=pelvis.ms_before_contact-shoulder.ms_before_contact;
        report.order=pelvis.uncertainty[1]<shoulder.uncertainty[0]?"Pelvis first":
            shoulder.uncertainty[1]<pelvis.uncertainty[0]?"Shoulders first":"Unresolved within frame uncertainty";
    }
    return report;
}

function objectReview(data,marks,context){
    const kind=shotKind(data.stroke), report={};
    const uncertain=contactProblem(data,marks);
    if(uncertain)return {reason:uncertain};
    if(!context.fixed)return {reason:"Confirm a fixed camera for projected measurements."};
    const [w,h]=data.image_size;
    const offset=(from,to)=>[(to[0]-from[0])*100,(from[1]-to[1])*h/w*100];
    const time=f=>data.times[data.frames.indexOf(f)];
    if(kind==="serve"){
        if(!marks.foot || !marks.release || !marks.apex || !marks.contact)
            return {reason:"Mark the setup front foot and ball at release, apex and contact."};
        if(!(marks.foot.frame<=marks.release.frame && marks.release.frame<marks.apex.frame && marks.apex.frame<marks.contact.frame && marks.contact.frame===data.impact))
            return {reason:"Verify setup → release → apex → selected contact."};
        return {apex_pct:offset(marks.foot.xy,marks.apex.xy),contact_pct:offset(marks.foot.xy,marks.contact.xy),
            apex_to_contact_ms:1000*(time(marks.contact.frame)-time(marks.apex.frame)),frames:[marks.release.frame,marks.apex.frame,data.impact]};
    }
    if(marks.contact && marks.contact.frame===data.impact && ["left","right"].includes(context.hand)){
        const shoulder=data.overlay[data.frames.indexOf(data.impact)][context.hand==="left"?11:12];
        if(shoulder)report.spacing_pct=offset(shoulder,marks.contact.xy);
    }
    if(marks.tip1 && marks.tip2 && marks.tip1.frame<marks.tip2.frame && marks.tip2.frame<=data.impact &&
        time(data.impact)-time(marks.tip1.frame)<=.15){
        const [dx,dy]=offset(marks.tip1.xy,marks.tip2.xy);
        if(Math.hypot(dx,dy)*w/100>=2)report.path_deg=Math.atan2(dy,Math.abs(dx))*180/Math.PI;
    }
    return report;
}

function shotReport(data,review){
    const {marks,context}=review;
    if(shotKind(data.stroke)==="forehand")
        return {rotation:rotationReview(data,marks,context),objects:objectReview(data,marks,context)};
    const result={objects:objectReview(data,marks,context)};
    for(const side of ["left","right"])
        result[side]=contactProblem(data,marks)?{reason:contactProblem(data,marks)}:marks.loading ? kneeExcursion(data,side,marks.loading.frame,data.impact) : {reason:"Mark the start of loading."};
    return result;
}

const GROUP_FIELDS=["player","setup","serveType","courtSide","target","lead","stance","hand"];
function tossGroup(reviews,current){
    if(GROUP_FIELDS.some(k=>!current.context[k] || current.context[k]==="unknown") || !current.context.fixed)return null;
    const seen=new Set(), matched=[];
    for(const review of [current,...reviews]){
        if(seen.has(review.shot_id) || review.stroke!=="serve" || !review.context.fixed ||
            !GROUP_FIELDS.every(k=>review.context[k]===current.context[k]))continue;
        const o=review.report.objects;
        if(![o.apex_pct,o.contact_pct].every(p=>Array.isArray(p) && p.length===2 && p.every(Number.isFinite)) ||
            !Number.isFinite(o.apex_to_contact_ms))continue;
        seen.add(review.shot_id);matched.push(o);
    }
    const stats=xs=>{const center=median(xs);return {median:center,mad:median(xs.map(x=>Math.abs(x-center)))}};
    if(matched.length<2)return {n:matched.length};
    const result={n:matched.length,apex_to_contact_ms:stats(matched.map(o=>o.apex_to_contact_ms))};
    for(const key of ["apex_pct","contact_pct"])
        result[key]=[0,1].map(axis=>stats(matched.map(o=>o[key][axis])));
    return result;
}

let review, peers=[], pendingPoint=null;
const kind=shotKind(data.stroke);
const fields=[
    ["player","Player",null], ["hand","Hitting hand",["unknown","left","right"]],
    ["setup","Camera setup ID",null], ["target","Intended target",null],
    ["outcome","Outcome",["unknown","in","long","wide","net"]],
    ["fixed","Fixed camera","check"], ["level","Fixed and level (camera Y vertical)","check"],
    ["groundStance","Forehand stance",["unknown","open","semi-open","neutral","closed"]],
    ["lead","Front foot",["unknown","left","right"]], ["stance","Serve stance",["unknown","foot-up","foot-back"]],
    ["serveType","Serve type",["unknown","flat","slice","kick"]], ["courtSide","Serving side",["unknown","deuce","ad"]]
];

function readEvidence(input){
    if(input.version!==1 || input.id!==data.review_id || input.stroke!==kind || !input.context || !input.marks)
        throw Error("Review belongs to a different stroke or has an unsupported format.");
    const clean={version:1,id:data.review_id,shot_id:data.shot_id,stroke:kind,context:{},marks:{}};
    for(const [key,,type] of fields){
        const value=input.context[key];
        clean.context[key]=type==="check"?value===true:type===null?String(value??"").trim().slice(0,100):
            type.includes(value)?value:"unknown";
    }
    for(const [key,mark] of Object.entries(input.marks)){
        if(!Object.hasOwn(SHOT_EVENTS[kind],key) || !mark || !data.frames.includes(mark.frame))
            throw Error("Unknown event or source frame.");
        if(key==="contact" && mark.frame!==data.impact)throw Error("Contact mark must match the selected impact.");
        const item={frame:mark.frame};
        if(POINT_EVENTS.has(key)){
            if(!Array.isArray(mark.xy) || mark.xy.length!==2 || !mark.xy.every(v=>Number.isFinite(v) && v>=0 && v<=1))
                throw Error("Invalid image position.");
            item.xy=mark.xy.slice();
        }
        clean.marks[key]=item;
    }
    return clean;
}

function setMessage(text){el("reviewStatus").textContent=text;}
function storeReview(){
    try{localStorage.setItem("reallyrally.review."+data.review_id,JSON.stringify(review));}
    catch {setMessage("Browser storage unavailable. Download the review to keep your marks.");}
    renderReview();overlay();
}
function stopMarking(){
    pendingPoint=null;el("pointHint").hidden=true;el("source").parentElement.classList.remove("marking");
}
function applyMark(event,item){
    review.marks[event]=item;stopMarking();setMessage("Saved "+SHOT_EVENTS[kind][event]+" at frame "+item.frame+".");storeReview();
}
function drawMarks(layer){
    if(!review)return;
    const [w,h]=data.image_size;
    layer.setAttribute("viewBox","0 0 "+w+" "+h);
    for(const item of Object.values(review.marks)){
        if(item.frame!==data.frames[frame] || !item.xy)continue;
        const dot=document.createElementNS("http://www.w3.org/2000/svg","circle");
        dot.setAttribute("cx",item.xy[0]*w);dot.setAttribute("cy",item.xy[1]*h);
        dot.setAttribute("r",Math.max(2,Math.max(w,h)*.008));dot.setAttribute("fill","none");
        dot.setAttribute("stroke","#ffed71");dot.setAttribute("stroke-width","2");
        layer.append(dot);
    }
}

function addFinding(title,text,frames=[]){
    const p=document.createElement("p"), strong=document.createElement("strong");
    strong.textContent=title+" · ";p.append(strong,document.createTextNode(text));
    for(const f of new Set(frames)){
        const button=document.createElement("button");button.textContent="Frame "+f;
        button.onclick=()=>seek(data.frames.indexOf(f));p.append(button);
    }
    el("findings").append(p);
}
const f1=n=>Number(n).toFixed(1);
const pair=p=>p.map(f1).join(", ");
function renderReview(){
    if(!review)return;
    const report=shotReport(data,review),o=report.objects;
    el("findings").replaceChildren();
    if(kind==="serve"){
        for(const side of ["left","right"]){
            const k=report[side],lead=review.context.lead;
            const name=side+" knee"+(["left","right"].includes(lead)?(lead===side?" (front)":" (rear)"):"");
            addFinding(name,k.reason??(f1(k.degrees)+"° excursion ("+f1(k.flexion_deg)+"° → "+f1(k.extension_deg)+"° flexion), most extended "+Math.round(k.ms_before_contact)+" ms before contact."),k.frames);
        }
        addFinding("Toss",o.reason??("Apex ("+pair(o.apex_pct)+"), contact ("+pair(o.contact_pct)+") relative to setup foot, % image width (right, up). "+Math.round(o.apex_to_contact_ms)+" ms apex to contact."),o.frames);
        const group=tossGroup(peers,{...review,report});
        if(group?.n>=2){
            const print=key=>group[key].map(v=>f1(v.median)+" ± "+f1(v.mad)).join(", ");
            addFinding("Matched serves",group.n+" repetitions. Median ± MAD, right/up in % image width. Apex: "+print("apex_pct")+". Contact: "+print("contact_pct")+". Apex to contact: "+Math.round(group.apex_to_contact_ms.median)+" ± "+Math.round(group.apex_to_contact_ms.mad)+" ms.");
        } else if(peers.length)addFinding("Matched serves","Need two complete, distinct reviews with matching player, fixed setup, hand, stance, serve type, court side and target.");
    }else{
        const r=report.rotation;
        if(r.reason)addFinding("Preparation and rotation",r.reason);
        else{
            for(const part of ["pelvis","shoulder"]){
                const prep=r[part+"_turn"],peak=r[part+"_peak"];
                addFinding(part+" turn",prep.reason??(f1(prep.degrees)+"° ready to end of preparation (forward positive)."),prep.frames);
                addFinding(part+" peak",peak.reason??(Math.round(peak.ms_before_contact)+" ms before impact."),peak.frames);
            }
            if(r.order)addFinding("Sequence",r.order+" · shoulder peak minus pelvis peak "+Math.round(r.interval_ms)+" ms.");
        }
        if(o.reason)addFinding("Contact and racket",o.reason);
        else{
            addFinding("Contact spacing",o.spacing_pct?"Ball minus hitting shoulder ("+pair(o.spacing_pct)+") % image width (right, up).":"Mark contact and identify a visible hitting shoulder.",o.spacing_pct?[data.impact]:[]);
            addFinding("Racket approach",Number.isFinite(o.path_deg)?f1(o.path_deg)+"° above screen horizontal; downward is negative.":"Mark two distinct tip positions in time order in the final 150 ms through impact.",Number.isFinite(o.path_deg)?[review.marks.tip1.frame,review.marks.tip2.frame]:[]);
        }
    }
    const matchKeys=kind==="serve"?GROUP_FIELDS:["player","setup","target","hand","groundStance"];
    if(peers.length && review.context.fixed && matchKeys.every(k=>review.context[k] && review.context[k]!=="unknown")){
        const unique=new Map(peers.filter(p=>p.stroke===kind && p.context.fixed && p.shot_id!==review.shot_id &&
            matchKeys.every(k=>review.context[k]===p.context[k])).map(p=>[p.shot_id,p]));
        const value=r=>kind==="serve"?r[review.context.lead]?.degrees:r.rotation?.shoulder_turn?.degrees;
        const baseline=[...unique.values()].map(p=>value(p.report)).filter(Number.isFinite);
        const now=value(report);
        if(Number.isFinite(now) && baseline.length){
            addFinding("Player comparison",(kind==="serve"?"Front-knee excursion":"Shoulder preparation turn")+" "+
                f1(now)+"° here; median "+f1(median(baseline))+"° in "+baseline.length+" other matched repetition(s).");
        }
    }
    el("markList").replaceChildren();
    for(const [key,mark] of Object.entries(review.marks)){
        const b=document.createElement("button");b.textContent=SHOT_EVENTS[kind][key]+" · "+mark.frame;
        b.onclick=()=>seek(data.frames.indexOf(mark.frame));el("markList").append(b);
    }
}

function contextControls(){
    for(const [key,label,type] of fields){
        if(kind!=="serve" && ["lead","stance","serveType","courtSide"].includes(key))continue;
        if(kind!=="forehand" && key==="groundStance")continue;
        if(kind==="serve" && key==="level")continue;
        const wrapper=document.createElement("label");wrapper.append(label+" ");
        const input=document.createElement(Array.isArray(type)?"select":"input");input.id="ctx-"+key;
        if(Array.isArray(type))for(const choice of type)input.add(new Option(choice,choice));
        else if(type==="check")input.type="checkbox";
        else {input.type="text";input.maxLength=100;}
        input.onchange=()=>{
            review.context[key]=type==="check"?input.checked:input.value.trim();
            if(key==="level" && input.checked){review.context.fixed=true;el("ctx-fixed").checked=true;}
            if(key==="fixed" && !input.checked && el("ctx-level")){review.context.level=false;el("ctx-level").checked=false;}
            storeReview();
        };
        wrapper.append(input);el("shotContext").append(wrapper);
    }
}
function syncContext(){
    for(const [key,,type] of fields){
        const input=el("ctx-"+key);if(!input)continue;
        if(type==="check")input.checked=review.context[key];else input.value=review.context[key];
    }
}

async function loadFiles(files){
    let next=review, imported=0;
    const others=new Map(peers.map(p=>[p.id,p]));
    try{
        for(const f of files){
            const candidate=JSON.parse(await f.text());
            if(candidate.id===data.review_id){next=readEvidence(candidate);imported++;continue;}
            if(candidate.version!==1 || typeof candidate.id!=="string" || typeof candidate.shot_id!=="string" || candidate.stroke!==kind ||
                !candidate.context || !candidate.report?.objects)throw Error("Expected a review exported for the same type of stroke.");
            others.set(candidate.id,candidate);imported++;
        }
        review=next;peers=[...others.values()];stopMarking();syncContext();setMessage("Loaded "+imported+" review(s).");storeReview();
    }catch(e){setMessage("Import failed: "+e.message);}
}
function startReview(){
    if(!Object.hasOwn(SHOT_EVENTS,kind))return;
    el("reviewPanel").hidden=false;
    el("reviewTitle").textContent=kind==="serve"?"Serve · loading and toss":"Forehand · preparation and delivery";
    review=readEvidence({version:1,id:data.review_id,stroke:kind,context:{hand:data.hitting_hand},marks:{}});
    try{
        const embedded=JSON.parse(el("initialReview").textContent);
        if(embedded)review=readEvidence(embedded);
        const saved=localStorage.getItem("reallyrally.review."+data.review_id);
        if(saved)review=readEvidence(JSON.parse(saved));
    }catch{setMessage("Use Download review to retain your annotations.");}
    contextControls();syncContext();
    for(const [key,label] of Object.entries(SHOT_EVENTS[kind]))el("eventType").add(new Option(label,key));
    el("eventType").onchange=stopMarking;
    el("markEvent").onclick=()=>{
        pause();const key=el("eventType").value;
        if(key==="contact")seek(impact);
        if(POINT_EVENTS.has(key)){
            pendingPoint=key;el("source").parentElement.classList.add("marking");
            el("pointHint").textContent="Click "+SHOT_EVENTS[kind][key]+" · Esc to cancel";
            el("pointHint").hidden=false;el("source").scrollIntoView({block:"center"});
            setMessage("Click "+SHOT_EVENTS[kind][key]+" on the recording at frame "+data.frames[frame]+".");
        }else applyMark(key,{frame:data.frames[frame]});
    };
    document.addEventListener("keydown",e=>{if(e.key==="Escape" && pendingPoint){stopMarking();setMessage("Mark cancelled.");}});
    el("source").onclick=e=>{
        if(!pendingPoint)return;
        const rect=el("source").getBoundingClientRect(),[w,h]=data.image_size;
        const scale=Math.min(rect.width/w,rect.height/h);
        const xy=[(e.clientX-rect.left-(rect.width-w*scale)/2)/(w*scale),(e.clientY-rect.top-(rect.height-h*scale)/2)/(h*scale)];
        if(xy.some(v=>v<0 || v>1))return;
        applyMark(pendingPoint,{frame:data.frames[frame],xy});
    };
    el("clearEvent").onclick=()=>{stopMarking();delete review.marks[el("eventType").value];setMessage("Mark cleared.");storeReview();};
    el("saveReview").onclick=()=>{
        const file={...review,report:shotReport(data,review)};
        const url=URL.createObjectURL(new Blob([JSON.stringify(file,null,2)],{type:"application/json"}));
        const a=document.createElement("a");a.href=url;a.download=kind+"-"+data.review_id+"-review.json";a.click();
        setTimeout(()=>URL.revokeObjectURL(url),1000);
    };
    el("reviewFiles").onchange=e=>{loadFiles([...e.target.files]);e.target.value="";};
    renderReview();
}
