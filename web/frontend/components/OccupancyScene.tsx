'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { PredictionRun } from '../lib/types';
import {boxCorners,type DetectionBox} from '../lib/detections';

type SceneState = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  mesh: THREE.InstancedMesh;
  points?: THREE.Points;
  boxes?: THREE.LineSegments;
  dirty: boolean;
  reset: () => void;
};

interface Props {
  run: Pick<PredictionRun,'bounds'|'voxel_size'|'classes'> & {occupancy?:{point_count:number}};
  voxels: Float32Array;
  lidar: Float32Array | null;
  showLidar: boolean;
  height: [number, number];
  resetVersion: number;
  onError: (message: string) => void;
  boxes?: DetectionBox[];
}

export default function OccupancyScene({ run, voxels, lidar, showLidar, height, resetVersion, onError,boxes=[] }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<SceneState | null>(null);

  useEffect(() => {
    const container = host.current;
    if (!container) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch (error) {
      onError(`Không khởi tạo được WebGL: ${String(error)}`);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute('aria-label', 'Occupancy 3D trong hệ tọa độ LiDAR');
    renderer.domElement.dataset.testid = 'occupancy-canvas';
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b1219');
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
    camera.up.set(0, 0, 1); // Preserve native LiDAR XYZ, with Z up.
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.maxDistance = 350;
    controls.minDistance = 5;
    controls.maxPolarAngle = Math.PI * 0.94;
    const reset = () => {
      // Drain residual OrbitControls damping before restoring the view. Without
      // this, a reset during/after dragging continues orbiting away from home.
      const damping = controls.enableDamping;
      controls.enableDamping = false;
      controls.update();
      const span = Math.max(run.bounds[3] - run.bounds[0], run.bounds[4] - run.bounds[1]);
      const x = (run.bounds[0] + run.bounds[3]) / 2;
      const y = (run.bounds[1] + run.bounds[4]) / 2;
      const fit = 1.13 / Math.min(1, camera.aspect);
      camera.position.set(x + span * 0.69 * fit, y - span * 0.82 * fit, span * 0.68 * fit);
      controls.target.set(x, y, (run.bounds[2] + run.bounds[5]) / 2);
      controls.update();
      controls.enableDamping = damping;
      if (state.current) state.current.dirty = true;
    };
    reset();
    scene.add(new THREE.HemisphereLight(0xe4f3ff, 0x667182, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(20, -35, 65);
    scene.add(sun);

    const grid = new THREE.GridHelper(Math.max(run.bounds[3] - run.bounds[0], run.bounds[4] - run.bounds[1]), 20, 0x33434f, 0x1b2a35);
    grid.rotation.x = Math.PI / 2;
    grid.position.x = (run.bounds[0] + run.bounds[3]) / 2;
    grid.position.y = (run.bounds[1] + run.bounds[4]) / 2;
    grid.position.z = run.bounds[2] - 0.03;
    scene.add(grid);
    const box = new THREE.Box3Helper(new THREE.Box3(
      new THREE.Vector3(...run.bounds.slice(0, 3)),
      new THREE.Vector3(...run.bounds.slice(3, 6)),
    ), new THREE.Color('#293b46'));
    scene.add(box);
    scene.add(new THREE.AxesHelper(10));

    const labelTextures: THREE.CanvasTexture[] = [];
    for (const [text, color, point] of [
      ['X', '#ff6868', [12, 0, 0]],
      ['Y', '#8ee191', [0, 12, 0]],
      ['Z', '#6baaff', [0, 0, 12]],
    ] as const) {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 64;
      const context = canvas.getContext('2d')!;
      context.font = 'bold 40px sans-serif';
      context.textAlign = 'center';
      context.fillStyle = color;
      context.fillText(text, 32, 47);
      const texture = new THREE.CanvasTexture(canvas);
      labelTextures.push(texture);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
      sprite.position.set(point[0], point[1], point[2]);
      sprite.scale.set(2.5, 2.5, 1);
      scene.add(sprite);
    }
    const geometry = new THREE.BoxGeometry(...run.voxel_size.map(value => value * 0.97) as [number, number, number]);
    const material = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, run.occupancy?.point_count || 0));
    mesh.count = 0;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    scene.add(mesh);
    const current: SceneState = { renderer, scene, camera, controls, mesh, dirty: true, reset };
    state.current = current;
    let firstResize = true;
    const resize = new ResizeObserver(() => {
      const width = container.clientWidth, height = container.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      if (firstResize) { reset(); firstResize = false; }
      current.dirty = true;
    });
    resize.observe(container);
    let animation = 0;
    const render = () => {
      animation = requestAnimationFrame(render);
      if (controls.update()) current.dirty = true;
      if (current.dirty) {
        renderer.render(scene, camera);
        renderer.domElement.dataset.renderedInstances = String(mesh.count);
        renderer.domElement.dataset.renderedPoints = String(current.points?.visible ? current.points.geometry.getAttribute('position').count : 0);
        renderer.domElement.dataset.renderedBoxes = String(current.boxes ? current.boxes.geometry.getAttribute('position').count / 24 : 0);
        renderer.domElement.dataset.webglReady = 'true';
        renderer.domElement.dataset.cameraPosition = camera.position.toArray().map(value => value.toFixed(3)).join(',');
        current.dirty = false;
      }
    };
    render();
    return () => {
      cancelAnimationFrame(animation);
      resize.disconnect();
      controls.dispose();
      scene.traverse(object => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.LineSegments) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach(value => value.dispose());
        } else if (object instanceof THREE.Sprite) object.material.dispose();
      });
      labelTextures.forEach(texture => texture.dispose());
      renderer.dispose();
      renderer.domElement.remove();
      state.current = null;
    };
  }, [run, onError]);

  useEffect(() => {
    const current = state.current;
    if (!current) return;
    const colors = new Map(run.classes.map(item => [item.id, new THREE.Color(`rgb(${item.color.join(',')})`)]));
    const matrix = new THREE.Matrix4();
    for (let i = 0; i < voxels.length; i += 5) {
      matrix.makeTranslation(voxels[i], voxels[i + 1], voxels[i + 2]);
      current.mesh.setMatrixAt(i / 5, matrix);
      current.mesh.setColorAt(i / 5, colors.get(voxels[i + 3])!);
    }
    current.mesh.count = voxels.length / 5;
    current.mesh.instanceMatrix.needsUpdate = true;
    if (current.mesh.instanceColor) current.mesh.instanceColor.needsUpdate = true;
    current.dirty = true;
  }, [run, voxels]);

  useEffect(() => {
    const current = state.current;
    if (!current) return;
    if (current.points) {
      current.scene.remove(current.points);
      current.points.geometry.dispose();
      (current.points.material as THREE.Material).dispose();
      current.points = undefined;
    }
    if (lidar && showLidar) {
      const positions: number[] = [];
      for (let i = 0; i < lidar.length; i += 4) {
        if (lidar[i + 2] >= height[0] && lidar[i + 2] <= height[1]) positions.push(lidar[i], lidar[i + 1], lidar[i + 2]);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      const material = new THREE.PointsMaterial({ color: '#d3edf2', size: 0.095, transparent: true, opacity: 0.85, sizeAttenuation: true });
      current.points = new THREE.Points(geometry, material);
      current.scene.add(current.points);
    }
    current.dirty = true;
  }, [run, lidar, showLidar, height]);

  useEffect(() => { state.current?.reset(); }, [resetVersion]);

  useEffect(()=>{
    const current=state.current;if(!current)return;
    if(current.boxes){current.scene.remove(current.boxes);current.boxes.geometry.dispose();(current.boxes.material as THREE.Material).dispose();current.boxes=undefined;}
    const vertices:number[]=[];const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    for(const box of boxes){const corners=boxCorners(box);for(const [a,b] of edges)vertices.push(...corners[a],...corners[b]);}
    if(vertices.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));current.boxes=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0xffcc45,depthTest:false}));current.boxes.renderOrder=10;current.scene.add(current.boxes);}
    current.dirty=true;
  },[run,boxes]);

  return <div className="scene-host" ref={host} />;
}
