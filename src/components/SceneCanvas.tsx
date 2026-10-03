'use client';

import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Bloom, EffectComposer, Vignette } from '@react-three/postprocessing';
import {
  AdditiveBlending,
  BackSide,
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  Group,
  InstancedBufferAttribute,
  InstancedMesh,
  Mesh,
  Object3D,
  PlaneGeometry,
  Points,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
} from 'three';
import { useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react';
import { useAppStore } from '@/state/store';

/* ════════════════════════════════════════════════════════════════
   LOOK KNOBS — tweak these first if you want it brighter / dimmer
   ════════════════════════════════════════════════════════════════ */
const SURFACE_Y = 5.4; // water surface height (seen from below)
const SEABED_Y = -8.5;
const SUN_DIR = new Vector3(0.19, 0.407, -0.893).normalize(); // direction of the sun in the sky
const SUN_POS = new Vector3(5.1, 17.6, -12); // where the sun glow sprite sits
const PITCH_RISE = 5.5; // camera looks upward by this much over 18 units (~17°)
const RAY_STRENGTH = 0.34; // god-ray brightness
const SUN_GLOW = 0.5; // sun halo brightness
const TINTS: [number, number, number][] = [
  [0.28, 0.78, 0.95],
  [0.45, 0.88, 0.95],
  [0.2, 0.62, 0.85],
  [0.62, 0.92, 0.92],
  [1.0, 0.62, 0.28], // amber accent fish
];

/* Reservoir mode (predictions page): the water level is a screen-space waterline.
   WATER.f  = animated fill, as a fraction of viewport height (matches the DOM gauge)
   WATER.on = 0..1 blend between "home/dams ceiling" and "reservoir waterline" */
const WATER = { f: 0.5, on: 0 };
const TAN_HALF = Math.tan((52 / 2) * (Math.PI / 180));
const RES_CAM = { y: 0.5, z: 12 };
/** World height of the waterline at depth z, so rays/bubbles/fish respect the screen-space waterline. */
const surfaceAt = (z: number) => {
  if (WATER.on < 0.001) return SURFACE_Y;
  const yw = RES_CAM.y + (WATER.f - 0.5) * 2 * TAN_HALF * (RES_CAM.z - z);
  return SURFACE_Y + (Math.max(-7.5, yw) - SURFACE_Y) * WATER.on;
};

function seeded(seed: number) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ════════════════════════════════════════════════════════════════
   SHARED GLSL
   ════════════════════════════════════════════════════════════════ */
const FINISH = `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

const PASS_VERT = `
  varying vec2 vUv;
  void main(){
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const NOISE = `
  float hash21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p){
    vec2 i = floor(p); vec2 f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
  }
`;

// Sum of directional waves. returns (height, dh/dx, dh/dz)
const WAVES = `
  void addWave(vec2 p, vec2 dir, float k, float a, float t, inout float h, inout vec2 g){
    float ph = dot(p, dir) * k + t * sqrt(9.81 * k) * 0.32;
    h += sin(ph) * a;
    g += dir * (k * a * cos(ph));
  }
  vec3 waves(vec2 p, float t, float f, float a){
    float h = 0.0; vec2 g = vec2(0.0);
    addWave(p, vec2( 0.966,  0.259), 0.50 * f, 0.160 * a, t, h, g);
    addWave(p, vec2(-0.342,  0.940), 0.83 * f, 0.110 * a, t, h, g);
    addWave(p, vec2( 0.707, -0.707), 1.37 * f, 0.070 * a, t, h, g);
    addWave(p, vec2(-0.866, -0.500), 2.30 * f, 0.045 * a, t, h, g);
    addWave(p, vec2( 0.259,  0.966), 3.90 * f, 0.026 * a, t, h, g);
    addWave(p, vec2(-0.940,  0.342), 6.40 * f, 0.014 * a, t, h, g);
    return vec3(h, g);
  }
`;

const CAUSTICS = `
  float caustics(vec2 uv, float time){
    vec2 p = mod(uv * 6.28318530718, 6.28318530718) - 250.0;
    vec2 i = p;
    float c = 1.0;
    float inten = 0.005;
    for (int n = 0; n < 4; n++) {
      float t = time * (1.0 - (3.5 / float(n + 1)));
      i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
      c += 1.0 / length(vec2(p.x / (sin(i.x + t) / inten), p.y / (cos(i.y + t) / inten)));
    }
    c /= 4.0;
    c = 1.17 - pow(c, 1.4);
    return pow(abs(c), 8.0);
  }
`;

/* ─────────── background dome (haze + soft sun halo) ─────────── */
const DOME_VERT = `
  varying vec3 vDir;
  void main(){
    vDir = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const DOME_FRAG = `
  varying vec3 vDir;
  uniform float uOpacity;
  uniform vec3 uSunDir;
  void main(){
    vec3 d = normalize(vDir);
    float y = d.y;
    vec3 col = vec3(0.0, 0.004, 0.008);
    col = mix(col, vec3(0.0, 0.055, 0.075), smoothstep(-0.6, 0.02, y));
    col = mix(col, vec3(0.008, 0.16, 0.21), smoothstep(0.0, 0.5, y));
    float q = y / 0.17;
    col += vec3(0.0, 0.09, 0.12) * exp(-q * q);
    float s = max(dot(d, uSunDir), 0.0);
    col += vec3(0.28, 0.82, 0.92) * (pow(s, 6.0) * 0.16 + pow(s, 40.0) * 0.4);
    gl_FragColor = vec4(col, uOpacity);
    ${FINISH}
  }
`;

/* ─────────── water ceiling, seen from below ─────────── */
const CEIL_VERT = `
  uniform float uTime;
  varying vec3 vWorld; varying vec3 vN; varying float vH;
  ${WAVES}
  void main(){
    vec3 p = position;
    vec3 w = waves(p.xz, uTime, 1.0, 1.0);
    p.y += w.x;
    vN = normalize(vec3(-w.y, 1.0, -w.z));
    vH = w.x;
    vec4 wp = modelMatrix * vec4(p, 1.0);
    vWorld = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;
const CEIL_FRAG = `
  uniform float uTime; uniform float uOpacity; uniform vec3 uSunDir;
  varying vec3 vWorld; varying vec3 vN; varying float vH;
  ${WAVES}
  void main(){
    // fine ripple detail on top of the big swells
    vec3 d = waves(vWorld.xz + vec2(3.7, 1.3), uTime * 1.25, 3.2, 0.28);
    vec3 N = normalize(vec3(vN.x - d.y, 1.0, vN.z - d.z));

    vec3 V = normalize(cameraPosition - vWorld);   // surface -> camera (downwards)
    vec3 Nf = -N;                                  // normal facing the camera
    vec3 I = -V;                                   // camera -> surface
    float cosT = clamp(dot(V, Nf), 0.0, 1.0);
    float dist = length(cameraPosition - vWorld);

    // Snell's window (widened on purpose so more of the sky shows)
    const float eta = 1.12;
    const float cC = 0.45;
    float win = smoothstep(cC - 0.06, cC + 0.12, cosT);

    vec3 Rt = refract(I, Nf, eta);
    Rt = (dot(Rt, Rt) < 0.01) ? vec3(0.0, 1.0, 0.0) : normalize(Rt);
    float up = clamp(Rt.y, 0.0, 1.0);
    vec3 sky = mix(vec3(0.08, 0.44, 0.56), vec3(0.42, 0.90, 1.0), pow(up, 1.6));
    float sd = max(dot(Rt, uSunDir), 0.0);
    vec3 sun = vec3(1.0, 0.97, 0.88) * (pow(sd, 260.0) * 5.0 + pow(sd, 30.0) * 0.45)
             + vec3(0.2, 0.8, 0.95) * pow(sd, 5.0) * 0.28;
    vec3 sky2 = sky * 0.8 + sun;

    // total internal reflection: a dark silvery mirror that picks up the ripples
    vec3 Rr = reflect(I, Nf);
    vec3 mirror = mix(vec3(0.015, 0.20, 0.25), vec3(0.0, 0.035, 0.05), smoothstep(0.0, 0.9, -Rr.y));
    mirror += vec3(0.04, 0.30, 0.36) * smoothstep(0.0, 0.5, length(N.xz)) * 0.9;

    vec3 col = mix(mirror, sky2, win);
    float q = (cosT - cC) / 0.045;
    col += vec3(0.25, 0.85, 1.0) * exp(-q * q) * 0.22;      // bright ring at the edge of the window
    col += vec3(0.10, 0.45, 0.55) * smoothstep(0.12, 0.34, vH) * 0.5; // wave crests

    col *= mix(0.72, 1.0, smoothstep(-14.0, 10.0, vWorld.x)); // keep the headline side calmer
    col *= mix(0.35, 1.0, exp(-dist * 0.010));
    float fade = 1.0 - smoothstep(34.0, 78.0, dist);

    gl_FragColor = vec4(col, uOpacity * fade * 0.96);
    ${FINISH}
  }
`;

/* ─────────── sun halo sprite ─────────── */
const GLOW_FRAG = `
  varying vec2 vUv; uniform float uOpacity;
  void main(){
    float r = length(vUv - 0.5) * 2.0;
    float g = exp(-r * r * 5.0) * 0.45 + exp(-r * r * 38.0) * 0.9;
    gl_FragColor = vec4(vec3(0.55, 0.95, 1.0), g * uOpacity);
  }
`;

/* ─────────── god rays ─────────── */
const RAY_VERT = `
  varying vec2 vUv; varying float vY;
  void main(){
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vY = wp.y;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;
const RAY_FRAG = `
  varying vec2 vUv; varying float vY;
  uniform float uTime; uniform float uOpacity; uniform float uSeed; uniform float uSurfaceY; uniform float uStrength;
  ${NOISE}
  void main(){
    float x = abs(vUv.x - 0.5) * 2.0;
    float core = exp(-x * x * 3.2);
    float below = clamp((uSurfaceY - vY) / 14.0, 0.0, 1.0);
    float above = max(vY - uSurfaceY, 0.0);
    float along = pow(1.0 - below, 1.7) * (1.0 - smoothstep(0.0, 6.0, above));
    float n = vnoise(vec2(vY * 0.35 - uTime * 0.25 + uSeed * 9.0, uSeed * 5.0 + uTime * 0.05));
    float n2 = vnoise(vec2(vUv.x * 3.0 + uSeed, vY * 0.18 + uTime * 0.12));
    float shimmer = 0.55 + 0.45 * n * (0.6 + 0.4 * n2);
    float a = core * along * shimmer * uStrength * uOpacity;
    gl_FragColor = vec4(vec3(0.36, 0.88, 1.0), a);
  }
`;

/* ─────────── caustic floor (seabed + tank bottom) ─────────── */
const FLOOR_VERT = `
  varying vec3 vWorld;
  void main(){
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorld = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;
const FLOOR_FRAG = `
  varying vec3 vWorld;
  uniform float uTime; uniform float uOpacity; uniform float uTiling; uniform float uFadeRate; uniform float uStrength;
  ${CAUSTICS}
  void main(){
    float d = length(cameraPosition.xz - vWorld.xz);
    float fade = exp(-pow(d * uFadeRate, 1.5));
    float c = caustics(vWorld.xz * uTiling, uTime);
    vec3 col = vec3(0.07, 0.62, 0.70) * c * uStrength;
    gl_FragColor = vec4(col, fade * uOpacity);
  }
`;

/* ─────────── marine snow (GPU points) ─────────── */
const SNOW_VERT = `
  attribute float aSeed; attribute float aSize;
  uniform float uTime; uniform float uScale; uniform float uMotion;
  varying float vAlpha;
  void main(){
    vec3 p = position;
    float t = uTime * uMotion;
    p.x += sin(t * 0.13 + aSeed * 40.0) * 0.35;
    p.y += sin(t * 0.17 + aSeed * 23.0) * 0.30;
    p.z += cos(t * 0.11 + aSeed * 31.0) * 0.30;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float dist = max(-mv.z, 0.1);
    gl_PointSize = clamp(aSize * uScale / dist, 1.0, 16.0);
    float q = dist * 0.045;
    vAlpha = exp(-q * q) * (0.55 + 0.45 * sin(t * 0.6 + aSeed * 90.0));
  }
`;
const SNOW_FRAG = `
  varying float vAlpha; uniform float uOpacity;
  void main(){
    float r = length(gl_PointCoord - 0.5) * 2.0;
    float a = 1.0 - smoothstep(0.0, 1.0, r);
    a *= a;
    gl_FragColor = vec4(vec3(0.45, 0.92, 1.0), a * vAlpha * uOpacity * 0.6);
  }
`;

/* ─────────── bubbles ─────────── */
const BUBBLE_VERT = `
  varying vec3 vN; varying vec3 vV;
  void main(){
    vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * mat3(instanceMatrix) * normal);
    vV = -mv.xyz;
    gl_Position = projectionMatrix * mv;
  }
`;
const BUBBLE_FRAG = `
  varying vec3 vN; varying vec3 vV; uniform float uOpacity;
  void main(){
    vec3 n = normalize(vN);
    vec3 v = normalize(vV);
    float ndv = clamp(dot(n, v), 0.0, 1.0);
    float rim = pow(1.0 - ndv, 2.2);
    vec3 l = normalize(vec3(-0.4, 0.8, 0.5));
    float spec = pow(max(dot(reflect(-l, n), v), 0.0), 40.0);
    float q = length(vV) * 0.035;
    float fog = exp(-q * q);
    float a = (rim * 0.8 + 0.03 + spec * 0.9) * uOpacity * fog;
    vec3 col = vec3(0.45, 0.92, 1.0) * rim + vec3(1.0) * spec + vec3(0.03, 0.12, 0.15);
    gl_FragColor = vec4(col, a);
  }
`;

/* ─────────── fish ─────────── */
const FISH_VERT = `
  attribute float aPhase; attribute vec3 aTint;
  uniform float uTime; uniform float uMotion;
  varying vec3 vTint; varying vec3 vN; varying vec3 vObj; varying vec3 vWorld;
  void main(){
    vec3 p = position;                                   // unit sphere, head at +z
    float tail = 1.0 - smoothstep(-1.0, 0.15, p.z);       // 0 head .. 1 tail
    float ped  = 1.0 - smoothstep(-0.70, -0.05, p.z);     // narrow tail stalk
    float fin  = 1.0 - smoothstep(-1.0, -0.72, p.z);      // tail fin
    p.xy *= mix(1.0, 0.45, ped);
    p.y  *= mix(1.0, 3.4, fin);
    p.x  *= mix(1.0, 0.25, fin);
    p.x  += sin(p.z * 3.0 - uTime * 7.0 + aPhase) * 0.55 * tail * tail * uMotion;

    mat4 M = modelMatrix * instanceMatrix;
    vec4 wp = M * vec4(p, 1.0);
    vWorld = wp.xyz;
    vN = normalize(transpose(inverse(mat3(M))) * normal);
    vObj = position;
    vTint = aTint;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;
const FISH_FRAG = `
  varying vec3 vTint; varying vec3 vN; varying vec3 vObj; varying vec3 vWorld;
  uniform float uTime; uniform float uOpacity;
  void main(){
    vec3 N = normalize(vN);
    vec3 V = normalize(cameraPosition - vWorld);
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float rim = pow(1.0 - ndv, 2.5);
    float top = N.y * 0.5 + 0.5;

    // dark back, silver belly
    float belly = 1.0 - smoothstep(-0.45, 0.15, vObj.y);
    vec3 base = mix(vTint * 0.55, mix(vTint, vec3(0.85, 0.97, 1.0), 0.55), belly);
    float c = 0.5 + 0.5 * sin(vWorld.x * 2.3 + vWorld.z * 1.7 + uTime * 1.3) * sin(vWorld.y * 2.9 - vWorld.x * 1.4 + uTime * 1.1);
    vec3 col = base * (0.35 + 0.65 * top) * (0.75 + 0.35 * c) + vec3(0.2, 0.8, 1.0) * rim * 0.45;

    // eyes
    float de = min(length(vObj - vec3(0.5, 0.25, 0.8)), length(vObj - vec3(-0.5, 0.25, 0.8)));
    float eye = 1.0 - smoothstep(0.10, 0.17, de);
    col = mix(col, vec3(0.02, 0.03, 0.04), eye);

    float dist = length(cameraPosition - vWorld);
    float q = dist * 0.038;
    float fog = exp(-q * q);
    col = mix(vec3(0.0, 0.05, 0.07), col, fog);
    gl_FragColor = vec4(col, uOpacity);
    ${FINISH}
  }
`;

/* ─────────── reservoir waterline (screen-space) ─────────── */
const MASK_VERT = `
  varying vec2 vUv;
  void main(){
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;
const MASK_FRAG = `
  varying vec2 vUv;
  uniform float uTime; uniform float uLevel; uniform float uOn; uniform float uAspect;
  void main(){
    float x = vUv.x * uAspect;
    float y = vUv.y;
    float w = sin(x * 3.1 + uTime * 0.55) * 0.0055
            + sin(x * 7.3 - uTime * 0.80) * 0.0030
            + sin(x * 17.0 + uTime * 1.50) * 0.0013;
    float d = y - (uLevel + w);                 // > 0 above the water
    float air = smoothstep(-0.0006, 0.0010, d);
    float q = d / 0.0022;
    float line = exp(-q * q);
    float sub = exp(-max(-d, 0.0) / 0.07) * (1.0 - air);
    float gx = (fract(uTime * 0.045) * 1.5 - 0.25) * uAspect;
    float gq = (x - gx) / 0.35;
    float glint = exp(-gq * gq);
    vec3 add = vec3(0.55, 0.95, 1.0) * line * (0.55 + 0.9 * glint)
             + vec3(0.06, 0.34, 0.42) * sub * (0.55 + 0.4 * glint);
    vec3 airCol = vec3(0.0, 0.010, 0.016) + vec3(0.02, 0.12, 0.15) * exp(-max(d, 0.0) / 0.05) * 0.6;
    gl_FragColor = vec4((airCol * air + add) * uOn, air * uOn);
  }
`;

type Presence = MutableRefObject<number>;

/* ════════════════════════════════════════════════════════════════
   UNDERWATER PIECES
   ════════════════════════════════════════════════════════════════ */
function Dome({ presence }: { presence: Presence }) {
  const camera = useThree((s) => s.camera);
  const mesh = useRef<Mesh>(null);
  const geometry = useMemo(() => new SphereGeometry(80, 32, 20), []);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: DOME_VERT,
        fragmentShader: DOME_FRAG,
        uniforms: { uOpacity: { value: 1 }, uSunDir: { value: SUN_DIR } },
        side: BackSide,
        transparent: true,
        depthWrite: false,
      }),
    [],
  );
  useFrame(() => {
    mesh.current?.position.copy(camera.position);
    material.uniforms.uOpacity.value = 0.15 + 0.85 * presence.current;
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <mesh ref={mesh} geometry={geometry} material={material} renderOrder={-100} frustumCulled={false} dispose={null} />;
}

function WaterSurface({ presence, reducedMotion, segments }: { presence: Presence; reducedMotion: boolean; segments: number }) {
  const mesh = useRef<Mesh>(null);
  const geometry = useMemo(() => {
    const g = new PlaneGeometry(150, 150, segments, segments);
    g.rotateX(-Math.PI / 2);
    return g;
  }, [segments]);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: CEIL_VERT,
        fragmentShader: CEIL_FRAG,
        uniforms: { uTime: { value: 0 }, uOpacity: { value: 1 }, uSunDir: { value: SUN_DIR } },
        transparent: true,
        depthWrite: false,
        side: DoubleSide,
      }),
    [],
  );
  useFrame((_, delta) => {
    const p = presence.current;
    if (mesh.current) mesh.current.visible = p > 0.01;
    material.uniforms.uTime.value += Math.min(delta, 0.05) * (reducedMotion ? 0.1 : 0.55);
    material.uniforms.uOpacity.value = p;
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <mesh ref={mesh} position={[0, SURFACE_Y, -25]} geometry={geometry} material={material} renderOrder={-50} frustumCulled={false} dispose={null} />;
}

function SunGlow({ presence }: { presence: Presence }) {
  const camera = useThree((s) => s.camera);
  const mesh = useRef<Mesh>(null);
  const geometry = useMemo(() => new PlaneGeometry(1, 1), []);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: PASS_VERT,
        fragmentShader: GLOW_FRAG,
        uniforms: { uOpacity: { value: SUN_GLOW } },
        transparent: true,
        depthWrite: false,
        depthTest: false,
        blending: AdditiveBlending,
      }),
    [],
  );
  useFrame(() => {
    const p = presence.current;
    if (!mesh.current) return;
    mesh.current.visible = p > 0.01;
    mesh.current.lookAt(camera.position);
    material.uniforms.uOpacity.value = SUN_GLOW * p;
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <mesh ref={mesh} position={SUN_POS} scale={[26, 26, 1]} geometry={geometry} material={material} renderOrder={20} frustumCulled={false} dispose={null} />;
}

function RayShafts({ presence, reducedMotion, count }: { presence: Presence; reducedMotion: boolean; count: number }) {
  const group = useRef<Group>(null);
  const geometry = useMemo(() => new PlaneGeometry(1, 1), []);
  const rays = useMemo(() => {
    const rnd = seeded(7);
    return Array.from({ length: count }, (_, i) => {
      const material = new ShaderMaterial({
        vertexShader: RAY_VERT,
        fragmentShader: RAY_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uOpacity: { value: 1 },
          uSeed: { value: rnd() * 10 },
          uSurfaceY: { value: SURFACE_Y },
          uStrength: { value: RAY_STRENGTH * (0.55 + rnd() * 0.6) },
        },
        transparent: true,
        depthWrite: false,
        side: DoubleSide,
        blending: AdditiveBlending,
      });
      return {
        material,
        x: SUN_POS.x * 0.35 + (rnd() - 0.5) * 46,
        z: -3 - rnd() * 22,
        width: 1.6 + rnd() * 4.2,
        tilt: -0.09 + (rnd() - 0.5) * 0.08,
        phase: i + rnd() * 6,
      };
    });
  }, [count]);
  const bottom = -10;
  const top = SURFACE_Y + 8;
  useFrame(({ clock }) => {
    const p = presence.current;
    if (group.current) group.current.visible = p > 0.01;
    const t = clock.elapsedTime * (reducedMotion ? 0.12 : 1);
    group.current?.children.forEach((child, i) => {
      const r = rays[i];
      if (!r) return;
      r.material.uniforms.uTime.value = t;
      r.material.uniforms.uOpacity.value = p;
      r.material.uniforms.uSurfaceY.value = surfaceAt(r.z);
      child.rotation.z = r.tilt + Math.sin(t * 0.16 + r.phase) * 0.02;
    });
  });
  useEffect(() => () => { geometry.dispose(); rays.forEach((r) => r.material.dispose()); }, [geometry, rays]);
  return (
    <group ref={group}>
      {rays.map((r, i) => (
        <mesh
          key={i}
          geometry={geometry}
          material={r.material}
          position={[r.x, (top + bottom) / 2, r.z]}
          scale={[r.width, top - bottom, 1]}
          renderOrder={10}
          frustumCulled={false}
          dispose={null}
        />
      ))}
    </group>
  );
}

function CausticFloor({
  position,
  size,
  tiling,
  fadeRate,
  strength,
  presence,
  reducedMotion,
}: {
  position: [number, number, number];
  size: [number, number];
  tiling: number;
  fadeRate: number;
  strength: number;
  presence: Presence;
  reducedMotion: boolean;
}) {
  const mesh = useRef<Mesh>(null);
  const geometry = useMemo(() => {
    const g = new PlaneGeometry(size[0], size[1]);
    g.rotateX(-Math.PI / 2);
    return g;
  }, [size]);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: FLOOR_VERT,
        fragmentShader: FLOOR_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uOpacity: { value: 1 },
          uTiling: { value: tiling },
          uFadeRate: { value: fadeRate },
          uStrength: { value: strength },
        },
        transparent: true,
        depthWrite: false,
        blending: AdditiveBlending,
      }),
    [tiling, fadeRate, strength],
  );
  useFrame((_, delta) => {
    const p = presence.current;
    if (mesh.current) mesh.current.visible = p > 0.01;
    material.uniforms.uTime.value += Math.min(delta, 0.05) * (reducedMotion ? 0.1 : 0.5);
    material.uniforms.uOpacity.value = p;
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <mesh ref={mesh} position={position} geometry={geometry} material={material} renderOrder={-40} frustumCulled={false} dispose={null} />;
}

function MarineSnow({ presence, amount, reducedMotion }: { presence: Presence; amount: number; reducedMotion: boolean }) {
  const ref = useRef<Points>(null);
  const geometry = useMemo(() => {
    const rnd = seeded(33);
    const pos = new Float32Array(amount * 3);
    const seed = new Float32Array(amount);
    const size = new Float32Array(amount);
    for (let i = 0; i < amount; i++) {
      pos[i * 3] = (rnd() - 0.5) * 44;
      pos[i * 3 + 1] = -8 + rnd() * 14;
      pos[i * 3 + 2] = 6 - rnd() * 32;
      seed[i] = rnd();
      size[i] = 0.03 + Math.pow(rnd(), 2.5) * 0.14;
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(pos, 3));
    g.setAttribute('aSeed', new BufferAttribute(seed, 1));
    g.setAttribute('aSize', new BufferAttribute(size, 1));
    return g;
  }, [amount]);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: SNOW_VERT,
        fragmentShader: SNOW_FRAG,
        uniforms: { uTime: { value: 0 }, uScale: { value: 1000 }, uMotion: { value: 1 }, uOpacity: { value: 1 } },
        transparent: true,
        depthWrite: false,
        blending: AdditiveBlending,
      }),
    [],
  );
  useFrame(({ clock, size, camera, gl }) => {
    const p = presence.current;
    if (ref.current) ref.current.visible = p > 0.01;
    const fov = ('fov' in camera ? (camera as { fov: number }).fov : 52) * (Math.PI / 180);
    material.uniforms.uScale.value = (size.height * gl.getPixelRatio() * 0.5) / Math.tan(fov * 0.5);
    material.uniforms.uTime.value = clock.elapsedTime;
    material.uniforms.uMotion.value = reducedMotion ? 0.2 : 1;
    material.uniforms.uOpacity.value = p;
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <points ref={ref} geometry={geometry} material={material} renderOrder={5} frustumCulled={false} dispose={null} />;
}

function ScrollBubbles({ amount, presence }: { amount: number; presence: Presence }) {
  const ref = useRef<InstancedMesh>(null);
  const dummy = useMemo(() => new Object3D(), []);
  const velocity = useRef(0);
  const geometry = useMemo(() => new SphereGeometry(1, 16, 12), []);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: BUBBLE_VERT,
        fragmentShader: BUBBLE_FRAG,
        uniforms: { uOpacity: { value: 1 } },
        transparent: true,
        depthWrite: false,
        blending: AdditiveBlending,
      }),
    [],
  );
  const bubbles = useMemo(() => {
    const rnd = seeded(5);
    return Array.from({ length: amount }, () => {
      const depth = rnd();
      return {
        x: (rnd() - 0.5) * 32,
        y: -8 + rnd() * 13,
        z: -1 - depth * 20,
        r: (0.05 + (1 - depth) * 0.13) * (0.6 + rnd() * 0.8),
        speed: 0.3 + (1 - depth) * 0.7,
        phase: rnd() * 10,
      };
    });
  }, [amount]);

  useFrame(({ clock }, delta) => {
    const mesh = ref.current;
    if (!mesh) return;
    const p = presence.current;
    mesh.visible = p > 0.01;
    if (!mesh.visible) return;
    const dt = Math.min(delta, 0.05);
    const target = Math.abs(useAppStore.getState().scrollVelocity);
    velocity.current += (target - velocity.current) * (1 - Math.exp(-dt * 12));
    const moving = velocity.current > 0.008; // bubbles only travel while the page is scrolling
    const t = clock.elapsedTime;
    material.uniforms.uOpacity.value = p;

    for (let i = 0; i < bubbles.length; i++) {
      const b = bubbles[i];
      const popTop = surfaceAt(b.z) - 0.05;
      if (moving) {
        b.y += dt * b.speed * (0.3 + velocity.current * 3.1);
        if (b.y > popTop) {
          b.y = -8;
          b.x = (Math.random() - 0.5) * 32;
        }
      }
      const fade = Math.max(0, Math.min(1, (popTop - b.y) / 1.0));
      const radius = b.r * fade * p;
      dummy.position.set(b.x + Math.sin(t * 0.8 + b.phase) * 0.04, b.y + Math.sin(t * 1.4 + b.phase) * 0.02, b.z);
      dummy.scale.setScalar(radius);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <instancedMesh ref={ref} args={[geometry, material, amount]} renderOrder={6} frustumCulled={false} />;
}

type Fish = { p: Vector3; v: Vector3; phase: number; size: number; school: number };
const _steer = new Vector3();
const _center = new Vector3();
const _align = new Vector3();
const _avoid = new Vector3();
const _diff = new Vector3();
const _look = new Vector3();

function FishSchool({ amount, presence, reducedMotion }: { amount: number; presence: Presence; reducedMotion: boolean }) {
  const ref = useRef<InstancedMesh>(null);
  const dummy = useMemo(() => new Object3D(), []);
  const fish = useMemo<Fish[]>(() => {
    const rnd = seeded(21);
    return Array.from({ length: amount }, (_, i) => ({
      p: new Vector3((rnd() - 0.5) * 22, -3.5 + rnd() * 6, -4 - rnd() * 13),
      v: new Vector3(rnd() - 0.5, (rnd() - 0.5) * 0.2, (rnd() - 0.5) * 0.4).setLength(0.8),
      phase: rnd() * 6.28,
      size: 0.7 + rnd() * 0.8,
      school: i < amount * 0.65 ? 0 : i + 1,
    }));
  }, [amount]);
  const geometry = useMemo(() => {
    const g = new SphereGeometry(1, 28, 20);
    g.rotateX(Math.PI / 2); // long axis -> +z, head at +z
    const phase = new Float32Array(amount);
    const tint = new Float32Array(amount * 3);
    fish.forEach((f, i) => {
      phase[i] = f.phase;
      const c = TINTS[i % TINTS.length];
      tint.set(c, i * 3);
    });
    g.setAttribute('aPhase', new InstancedBufferAttribute(phase, 1));
    g.setAttribute('aTint', new InstancedBufferAttribute(tint, 3));
    return g;
  }, [amount, fish]);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: FISH_VERT,
        fragmentShader: FISH_FRAG,
        uniforms: { uTime: { value: 0 }, uMotion: { value: 1 }, uOpacity: { value: 1 } },
        transparent: true,
      }),
    [],
  );

  useFrame(({ clock }, delta) => {
    const mesh = ref.current;
    if (!mesh) return;
    const p = presence.current;
    mesh.visible = p > 0.01 && amount > 0;
    if (!mesh.visible) return;
    const dt = Math.min(delta, 0.05);
    const t = clock.elapsedTime;
    const motion = reducedMotion ? 0.25 : 1;
    const scrollV = useAppStore.getState().scrollVelocity;
    material.uniforms.uTime.value = t;
    material.uniforms.uMotion.value = motion;
    material.uniforms.uOpacity.value = p;

    for (let i = 0; i < fish.length; i++) {
      const f = fish[i];
      _steer.set(0, 0, 0);
      _center.set(0, 0, 0);
      _align.set(0, 0, 0);
      _avoid.set(0, 0, 0);
      let near = 0;
      for (let j = 0; j < fish.length; j++) {
        if (j === i) continue;
        const o = fish[j];
        _diff.subVectors(f.p, o.p);
        const d2 = _diff.lengthSq();
        if (d2 < 2.2) _avoid.addScaledVector(_diff, 1 / Math.max(0.2, d2));
        if (f.school === o.school && d2 < 20) {
          _center.add(o.p);
          _align.add(o.v);
          near++;
        }
      }
      if (near) {
        _center.multiplyScalar(1 / near).sub(f.p);
        _align.multiplyScalar(1 / near).sub(f.v);
        _steer.addScaledVector(_center, 0.22).addScaledVector(_align, 0.9);
      }
      _steer.addScaledVector(_avoid, 0.5);
      // lazy wander
      _steer.x += Math.sin(t * 0.31 * motion + f.phase) * 0.18 * motion;
      _steer.y += Math.cos(t * 0.4 * motion + f.phase) * 0.1 * motion;
      _steer.z += Math.sin(t * 0.23 * motion + f.phase * 2) * 0.12 * motion;
      // soft bounds
      if (Math.abs(f.p.x) > 12) _steer.x -= Math.sign(f.p.x) * 1.6;
      const topLimit = Math.min(3.2, surfaceAt(f.p.z) - 0.7);
      const bottomLimit = Math.max(-7.2, Math.min(-4.8, topLimit - 1.2));
      if (f.p.y > topLimit) _steer.y -= 1.6 + (f.p.y - topLimit);
      if (f.p.y < bottomLimit) _steer.y += 1.2;
      if (f.p.z > -3) _steer.z -= 1.4;
      if (f.p.z < -17) _steer.z += 1.4;
      // startle when the page is being scrolled hard
      if (scrollV > 0.8) {
        _steer.y += Math.min(1.2, scrollV * 0.4);
        _steer.z -= Math.min(1.0, scrollV * 0.3);
      }

      f.v.addScaledVector(_steer, dt * 1.4);
      const maxS = (reducedMotion ? 0.5 : 1.45) * (1 + Math.min(scrollV, 2) * 0.2);
      f.v.clampLength(reducedMotion ? 0.15 : 0.5, maxS);
      f.p.addScaledVector(f.v, dt);

      dummy.position.copy(f.p);
      dummy.lookAt(_look.copy(f.p).add(f.v));
      dummy.scale.set(f.size * 0.17, f.size * 0.22, f.size * 0.55);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  if (amount === 0) return null;
  return <instancedMesh ref={ref} args={[geometry, material, amount]} renderOrder={0} frustumCulled={false} />;
}

/* ════════════════════════════════════════════════════════════════
   RESERVOIR WATERLINE (predictions page)
   Everything above the waterline is painted dark; the line itself shimmers.
   The underwater world behind it is the exact same scene as Home / Dams.
   ════════════════════════════════════════════════════════════════ */
function ReservoirMask({ presence, reducedMotion }: { presence: Presence; reducedMotion: boolean }) {
  const mesh = useRef<Mesh>(null);
  const geometry = useMemo(() => new PlaneGeometry(2, 2), []);
  const material = useMemo(
    () =>
      new ShaderMaterial({
        vertexShader: MASK_VERT,
        fragmentShader: MASK_FRAG,
        uniforms: { uTime: { value: 0 }, uLevel: { value: 0.5 }, uOn: { value: 0 }, uAspect: { value: 1.8 } },
        transparent: true,
        premultipliedAlpha: true,
        depthTest: false,
        depthWrite: false,
      }),
    [],
  );
  useFrame(({ clock, size }) => {
    const p = presence.current;
    if (mesh.current) mesh.current.visible = p > 0.01;
    material.uniforms.uTime.value = clock.elapsedTime * (reducedMotion ? 0.15 : 1);
    material.uniforms.uLevel.value = WATER.f;
    material.uniforms.uOn.value = p;
    material.uniforms.uAspect.value = size.width / Math.max(1, size.height);
  });
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  return <mesh ref={mesh} geometry={geometry} material={material} renderOrder={100} frustumCulled={false} dispose={null} />;
}

/* ════════════════════════════════════════════════════════════════
   SCENE
   ════════════════════════════════════════════════════════════════ */
function SceneContent({ visible }: { visible: boolean }) {
  const mode = useAppStore((s) => s.scene);
  const quality = useAppStore((s) => s.quality);
  const [reducedMotion, setReducedMotion] = useState(false);
  const camera = useThree((s) => s.camera);
  const scroll = useRef(0);
  const mouse = useRef({ x: 0, y: 0 });
  const world = useRef(1); // the underwater world (home, dams and predictions)
  const ceiling = useRef(1); // from-below water ceiling + sun halo (home / dams)
  const reservoir = useRef(0); // screen-space waterline (predictions)
  const look = useRef(new Vector3(0, 6, -6));
  const goal = useMemo(() => new Vector3(), []);
  const home = mode === 'underwater';
  const res = mode === 'tank';

  useFrame((_, delta) => {
    if (!visible) return;
    const dt = Math.min(delta, 0.05);
    const k = Math.min(1, dt * 2.4);
    world.current += ((home || res ? 1 : 0) - world.current) * k;
    ceiling.current += ((home ? 1 : 0) - ceiling.current) * k;
    reservoir.current += ((res ? 1 : 0) - reservoir.current) * k;

    // animated waterline: follows the reservoir fill (or the predicted fill when that toggle is on)
    const s = useAppStore.getState();
    const target = Math.max(0.015, Math.min(0.985, s.showPredicted ? s.predictedFraction : s.tankFraction));
    if (reservoir.current < 0.02) WATER.f = target;
    else WATER.f += (target - WATER.f) * (1 - Math.exp(-dt * 2.4));
    WATER.on = reservoir.current;

    // home/dams: camera tilts up at the ceiling and dives with scroll.
    // predictions: camera is level, so screen height == waterline height.
    const dive = res ? 0 : Math.min(scroll.current * 0.0016, 3.0);
    const camX = mouse.current.x * (res ? 0.3 : 0.5);
    const camY = res ? RES_CAM.y : 0.5 - dive - mouse.current.y * 0.2;
    camera.position.x += (camX - camera.position.x) * Math.min(1, dt * 1.4);
    camera.position.y += (camY - camera.position.y) * Math.min(1, dt * 1.4);
    goal.set(mouse.current.x * (res ? 0.6 : 1.4), camY + PITCH_RISE * (1 - reservoir.current) - (res ? 0 : mouse.current.y * 0.5), -6);
    look.current.lerp(goal, Math.min(1, dt * 1.8));
    camera.lookAt(look.current);
  });

  useEffect(() => {
    let previousY = window.scrollY;
    let previousAt = performance.now();
    let decayRaf = 0;
    let decay = 0;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const onScroll = () => {
      scroll.current = window.scrollY;
      const now = performance.now();
      const dt = Math.max(12, now - previousAt);
      const dy = window.scrollY - previousY;
      previousY = window.scrollY;
      previousAt = now;
      if (reduceMotion) return;
      decay = Math.min(3.2, (Math.abs(dy) / dt) * 0.48);
      useAppStore.getState().setScrollVelocity(decay);
      cancelAnimationFrame(decayRaf);
      const tick = () => {
        decay *= 0.86;
        if (decay < 0.045) {
          decay = 0;
          useAppStore.getState().setScrollVelocity(0);
          return;
        }
        useAppStore.getState().setScrollVelocity(decay);
        decayRaf = requestAnimationFrame(tick);
      };
      decayRaf = requestAnimationFrame(tick);
    };
    const onMouse = (e: MouseEvent) => {
      mouse.current.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouse.current.y = (e.clientY / window.innerHeight - 0.5) * 2;
    };
    scroll.current = window.scrollY;
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('mousemove', onMouse, { passive: true });
    return () => {
      cancelAnimationFrame(decayRaf);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('mousemove', onMouse);
    };
  }, []);

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReducedMotion(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  const hi = quality === 'high';
  const mid = quality === 'medium';

  return (
    <>
      <color attach="background" args={['#000000']} />
      <Dome presence={world} />
      <WaterSurface presence={ceiling} reducedMotion={reducedMotion} segments={hi ? 240 : mid ? 160 : 100} />
      <SunGlow presence={ceiling} />
      <CausticFloor position={[0, SEABED_Y, -30]} size={[140, 140]} tiling={0.11} fadeRate={0.03} strength={0.9} presence={world} reducedMotion={reducedMotion} />
      <MarineSnow presence={world} reducedMotion={reducedMotion} amount={hi ? 900 : mid ? 450 : 200} />
      <FishSchool presence={world} reducedMotion={reducedMotion} amount={hi ? 12 : mid ? 7 : 4} />
      <ScrollBubbles presence={world} amount={hi ? 360 : mid ? 190 : 80} />
      <RayShafts presence={world} reducedMotion={reducedMotion} count={hi ? 16 : mid ? 11 : 7} />
      <ReservoirMask presence={reservoir} reducedMotion={reducedMotion} />
      {hi && (
        <EffectComposer multisampling={4}>
          <Bloom intensity={0.55} luminanceThreshold={0.7} luminanceSmoothing={0.3} mipmapBlur />
          <Vignette eskil={false} offset={0.18} darkness={0.65} />
        </EffectComposer>
      )}
    </>
  );
}

export default function SceneCanvas() {
  const [isVisible, setIsVisible] = useState(true);
  const [devOpen, setDevOpen] = useState(false);
  const quality = useAppStore((s) => s.quality);
  const setQuality = useAppStore((s) => s.setQuality);
  const manual = useRef(false);

  useEffect(() => {
    const onVisibility = () => setIsVisible(document.visibilityState === 'visible');
    const onKey = (event: KeyboardEvent) => {
      if (event.altKey && event.shiftKey && event.code === 'KeyQ') {
        event.preventDefault();
        setDevOpen((value) => !value);
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('keydown', onKey);

    // FPS probe: skip the first 2.5s (shader compile hitches), then take the median frame time.
    let raf = 0;
    let last = 0;
    const start = performance.now();
    const samples: number[] = [];
    const probe = (now: number) => {
      if (last && now - start > 2500) samples.push(now - last);
      last = now;
      if (now - start < 5500) {
        raf = requestAnimationFrame(probe);
        return;
      }
      if (manual.current || samples.length < 20) return;
      samples.sort((a, b) => a - b);
      const fps = 1000 / samples[Math.floor(samples.length / 2)];
      setQuality(fps < 20 ? 'low' : fps < 38 ? 'medium' : 'high');
    };
    raf = requestAnimationFrame(probe);

    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('keydown', onKey);
      cancelAnimationFrame(raf);
    };
  }, [setQuality]);

  return (
    <div className="scene-canvas" aria-hidden="true">
      <Canvas
        frameloop={isVisible ? 'always' : 'never'}
        dpr={[1, 1.75]}
        camera={{ position: [0, 0.5, 12], fov: 52, near: 0.1, far: 140 }}
        gl={{ antialias: false, alpha: false, powerPreference: 'high-performance' }}
        onCreated={({ gl }) => {
          gl.setClearColor(new Color('#000000'), 1);
          gl.outputColorSpace = 'srgb';
          gl.toneMappingExposure = 1.1;
        }}
      >
        <SceneContent visible={isVisible} />
      </Canvas>
      <div className={`scene-dev ${devOpen ? 'open' : ''}`}>
        <button
          type="button"
          onClick={() => {
            manual.current = true;
            const q = useAppStore.getState().quality;
            setQuality(q === 'high' ? 'medium' : q === 'medium' ? 'low' : 'high');
          }}
          tabIndex={devOpen ? 0 : -1}
          title={`Quality: ${quality}`}
        >
          • {quality} quality
        </button>
        <span>Alt + Shift + Q</span>
      </div>
    </div>
  );
}