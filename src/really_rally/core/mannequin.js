// Render the exported GLB itself, with continuous pose tracks and hard visibility gates, on a 2D canvas.
// A small depth buffer keeps crossed limbs correctly occluded even without WebGL.
class Mannequin {
    constructor(encodedGLB) {
        const bytes = Uint8Array.from(atob(encodedGLB), char => char.charCodeAt(0));
        const jsonLength = new DataView(bytes.buffer).getUint32(12, true);
        const gltf = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
        const binaryStart = 28 + jsonLength;
        function accessor(id) {
            const spec = gltf.accessors[id], view = gltf.bufferViews[spec.bufferView];
            const type = spec.componentType === 5123 ? Uint16Array : Float32Array;
            return new type(bytes.buffer, binaryStart + view.byteOffset, view.byteLength / type.BYTES_PER_ELEMENT);
        }
        const meshes = gltf.meshes.map(({primitives: [p]}) => ({
            position: accessor(p.attributes.POSITION), normal: accessor(p.attributes.NORMAL), index: accessor(p.indices)
        }));
        const tracks = gltf.nodes.map(() => ({}));
        const animation = gltf.animations[0];
        this.times = accessor(animation.samplers[0].input);
        for (const {sampler, target} of animation.channels) {
            tracks[target.node][target.path] = accessor(animation.samplers[sampler].output);
        }
        const gates = new Map();
        gltf.nodes.forEach((node, i) => {
            if (node.children) gates.set(node.children[0], tracks[i].scale);
        });
        this.parts = gltf.nodes.flatMap((node, i) => node.mesh === undefined ? [] :
            [{mesh: meshes[node.mesh], ...tracks[i], gate: gates.get(i)}]);
        const maxVertices = Math.max(...meshes.map(mesh => mesh.position.length / 3));
        this.vertices = new Float32Array(4 * maxVertices);
        this.surface = document.createElement("canvas");
        this.context = this.surface.getContext("2d");
    }

    sample(time) {
        time = Math.fround(time);
        let lo = 0, hi = this.times.length;
        while (lo + 1 < hi) {
            const mid = (lo + hi) >> 1;
            if (this.times[mid] <= time) lo = mid; else hi = mid;
        }
        const next = Math.min(lo+1, this.times.length-1);
        const mix = next === lo ? 0 : Math.max(0, Math.min(1, (time-this.times[lo])/(this.times[next]-this.times[lo])));
        return {lo, next, mix};
    }

    quaternion(track, lo, next, mix) {
        const a = track.subarray(4*lo, 4*lo+4);
        if (!mix) return a;
        const b = track.subarray(4*next, 4*next+4);
        const dot = a.reduce((sum,v,k) => sum + v*b[k], 0);
        const sign = dot < 0 ? -1 : 1;
        const theta = Math.acos(Math.min(1, Math.abs(dot)));
        const weight = t => theta < .001 ? t : Math.sin(t*theta)/Math.sin(theta);
        const wa = weight(1-mix), wb = sign*weight(mix);
        const result = a.map((v,k) => wa*v + wb*b[k]);
        const length = Math.hypot(...result);
        return result.map(v => v/length);
    }

    draw(ctx, time, width, height, scale, yaw, pitch, cameraY) {
        const density = Math.min(1.65, 1000 / Math.max(width, height));
        const w = Math.max(1, Math.round(width * density)), h = Math.max(1, Math.round(height * density));
        if (!this.image || this.surface.width !== w || this.surface.height !== h) {
            this.surface.width = w;
            this.surface.height = h;
            this.image = this.context.createImageData(w, h);
            this.depth = new Float32Array(w * h);
        }
        this.depth.fill(-Infinity);
        this.image.data.fill(0);
        const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
        const size = scale * density;
        let visible = 0;
        const {lo, next, mix} = this.sample(time);
        const value = (track, axis) => (1-mix)*track[3*lo+axis] + mix*track[3*next+axis];
        for (const {mesh, translation, rotation, scale: localScale, gate} of this.parts) {
            if (gate && !gate[3*lo]) continue;
            const sx = value(localScale,0), syPart = value(localScale,1), sz = value(localScale,2);
            if (!sx || !syPart || !sz) continue;
            visible++;
            const [qx, qy, qz, qw] = this.quaternion(rotation, lo, next, mix);
            // Columns of the local rotation, expressed in camera coordinates.
            const axes = [
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy + qw*qz), 2*(qx*qz - qw*qy)],
                [2*(qx*qy - qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz + qw*qx)],
                [2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx*qx + qy*qy)],
                [value(translation,0), value(translation,1)-cameraY, value(translation,2)]
            ].map(([x,y,z]) => {
                const depth = -sy*x + cy*z;
                return [cy*x + sy*z, cp*y - sp*depth, cp*depth + sp*y];
            });
            const [a, b, c, origin] = axes;
            for (let k = 0, v = 0; k < mesh.position.length; k += 3, v += 4) {
                const x = mesh.position[k]*sx, y = mesh.position[k+1]*syPart, z = mesh.position[k+2]*sz;
                this.vertices[v] = w/2 + size*(origin[0] + a[0]*x + b[0]*y + c[0]*z);
                this.vertices[v+1] = h/2 - size*(origin[1] + a[1]*x + b[1]*y + c[1]*z);
                this.vertices[v+2] = origin[2] + a[2]*x + b[2]*y + c[2]*z;
                // Normals use inverse scale so long, narrow parts light correctly.
                const u = mesh.normal[k]/sx, t = mesh.normal[k+1]/syPart, r = mesh.normal[k+2]/sz;
                const length = Math.hypot(u,t,r);
                const nx = (a[0]*u + b[0]*t + c[0]*r)/length;
                const ny = (a[1]*u + b[1]*t + c[1]*r)/length;
                const nz = (a[2]*u + b[2]*t + c[2]*r)/length;
                const key = Math.max(0, -.44*nx + .72*ny + .54*nz);
                const gloss = Math.max(0, -.25*nx + .41*ny + .875*nz) ** 18;
                const rim = Math.max(0, .62*nx + .44*ny - .62*nz);
                this.vertices[v+3] = Math.min(1, .09 + .77*key + .24*gloss + .26*rim);
            }
            for (let k = 0; k < mesh.index.length; k += 3) {
                this.triangle(mesh.index[k]*4, mesh.index[k+1]*4, mesh.index[k+2]*4, w, h);
            }
        }
        this.context.putImageData(this.image, 0, 0);
        ctx.drawImage(this.surface, 0, 0, width, height);
        return visible;
    }

    triangle(a, b, c, w, h) {
        const v = this.vertices, pixels = this.image.data;
        const ax=v[a], ay=v[a+1], bx=v[b], by=v[b+1], cx=v[c], cy=v[c+1];
        const det = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
        if (det >= -0.00001) return;
        const left=Math.max(0, Math.ceil(Math.min(ax,bx,cx)-.5)), right=Math.min(w-1, Math.floor(Math.max(ax,bx,cx)-.5));
        const top=Math.max(0, Math.ceil(Math.min(ay,by,cy)-.5)), bottom=Math.min(h-1, Math.floor(Math.max(ay,by,cy)-.5));
        const db = (cy-ay)/det, dc = (ay-by)/det;
        for (let y=top; y<=bottom; y++) {
            let wb = db*(left+.5-ax) - (cx-ax)/det*(y+.5-ay);
            let wc = dc*(left+.5-ax) + (bx-ax)/det*(y+.5-ay);
            for (let x=left, i=y*w+left; x<=right; x++, i++, wb+=db, wc+=dc) {
                const wa = 1-wb-wc;
                if (Math.min(wa,wb,wc) < 0) continue;
                const z = wa*v[a+2] + wb*v[b+2] + wc*v[c+2];
                if (z <= this.depth[i]) continue;
                this.depth[i] = z;
                const light = wa*v[a+3] + wb*v[b+3] + wc*v[c+3];
                pixels[i*4] = 102 + 153*light;
                pixels[i*4+1] = 127 + 127*light;
                pixels[i*4+2] = 148 + 103*light;
                pixels[i*4+3] = 255;
            }
        }
    }
}
