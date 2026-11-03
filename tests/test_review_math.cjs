// Run with node --test tests/test_review_math.cjs (no browser or npm dependencies).
const {test}=require("node:test");
const assert=require("node:assert/strict");
const {readFileSync}=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const data={stroke:"serve",frames:Array.from({length:33},(_,i)=>40+i),impact:69,image_size:[800,400]};
data.times=data.frames.map((_,i)=>i*.02);
const api={data};
vm.runInNewContext(readFileSync(path.join(__dirname,"../src/really_rally/core/shot_review.js"),"utf8"),api);

test("excursion uses ordered extrema and withholds a leg with a gap",()=>{
    const motion={...data,kinematics:data.frames.map((f,i)=>({left_knee_flexion_deg:[20,30,65,40,15,22][i]}))};
    const k=api.kneeExcursion(motion,"left",40,45);
    assert.equal(k.degrees,50);assert.deepEqual(Array.from(k.frames),[42,44,45]);
    assert.ok(Math.abs(k.ms_before_contact-20)<1e-8);
    motion.kinematics[1].left_knee_flexion_deg=null;
    assert.equal(api.kneeExcursion(motion,"left",40,45).frames[0],41);
});

test("forehand turn unwraps headings; identifiable forward peaks remain separate",()=>{
    const motion={...data,stroke:"forehand"};
    let pelvis=170, shoulder=170;
    const wrap=x=>(x+540)%360-180;
    motion.kinematics=data.frames.map((f,i)=>{
        const record={pelvis_heading_deg:wrap(pelvis),shoulder_heading_deg:wrap(shoulder)};
        pelvis+=i<5?3:10*Math.exp(-((i-12)**2)/6);
        shoulder+=i<5?4:12*Math.exp(-((i-20)**2)/6);
        return record;
    });
    const marks={ready:{frame:40},prepared:{frame:45}};
    const r=api.rotationReview(motion,marks,{level:true,hand:"right"});
    assert.ok(Math.abs(r.pelvis_turn.degrees-15)<1e-8);
    assert.equal(r.order,"Pelvis first");assert.ok(r.interval_ms>100);
    for(const row of motion.kinematics)row.shoulder_heading_deg=row.pelvis_heading_deg;
    assert.match(api.rotationReview(motion,marks,{level:true,hand:"right"}).order,/Unresolved/);
    marks.early={frame:68};marks.late={frame:70};
    assert.match(api.rotationReview(motion,marks,{level:true,hand:"right"}).pelvis_peak.reason,/uncertain/);
});

test("matched toss stats use distinct shots, same conditions and actual image aspect",()=>{
    const context=Object.fromEntries(["setup","player","target","serveType","courtSide","stance","hand","lead"].map(k=>[k,"test"]));
    context.fixed=true;
    const marks={foot:{frame:40,xy:[.5,.9]},release:{frame:44,xy:[.5,.6]},apex:{frame:52,xy:[.6,.1]},contact:{frame:69,xy:[.55,.2]}};
    const o=api.objectReview(data,marks,context);
    assert.ok(Math.abs(o.apex_pct[0]-10)<1e-8);assert.equal(o.apex_pct[1],40);
    const current={stroke:"serve",context,shot_id:"a",report:{objects:o}};
    const peers=[{...current,shot_id:"b"},current,{...current,shot_id:"c",context:{...context,target:"elsewhere"}}];
    const group=api.tossGroup(peers,current);assert.equal(group.n,2);assert.equal(group.apex_pct[1].mad,0);
});

test("spacing uses the chosen shoulder and approach ignores a late or stationary tip pair",()=>{
    const motion={...data,stroke:"forehand",overlay:data.frames.map(()=>Array.from({length:33},()=>[.4,.7]))};
    const ctx={fixed:true,hand:"left"};
    const marks={contact:{frame:69,xy:[.6,.5]},tip1:{frame:66,xy:[.3,.8]},tip2:{frame:69,xy:[.4,.6]}};
    const o=api.objectReview(motion,marks,ctx);assert.ok(Math.abs(o.spacing_pct[0]-20)<1e-8);
    assert.ok(Math.abs(o.path_deg-45)<1e-8);
    marks.tip1.frame=40;assert.equal(api.objectReview(motion,marks,ctx).path_deg,undefined);
});
