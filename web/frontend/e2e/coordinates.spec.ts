import { expect, test } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { apiLogin, pageLogin } from './auth-helpers';

const root = path.resolve(__dirname, '../../..');
const metadataPath = path.join(root, 'work_dirs/infer_mini_cuda_v2/metadata.json');

test('requested asymmetric voxel center stays native XYZ in three.js mesh and point geometry', () => {
  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  const index = [10, 30, 4];
  const center = ['x', 'y', 'z'].map((axis, worldAxis) =>
    metadata.bounds[worldAxis] + (index[metadata.grid_axis_order.indexOf(axis)] + .5) * metadata.voxel_size[worldAxis]);
  expect(center).toEqual([-44.75, -34.75, -2.75]);

  // This is an isolated control point, not a fabricated prediction artifact.
  // Check the three.js matrix/storage conventions used by OccupancyScene.
  const scene = new THREE.Scene();
  const box = new THREE.BoxGeometry(...metadata.voxel_size as [number, number, number]);
  const boxMaterial = new THREE.MeshBasicMaterial();
  const mesh = new THREE.InstancedMesh(box, boxMaterial, 1);
  mesh.setMatrixAt(0, new THREE.Matrix4().makeTranslation(center[0], center[1], center[2]));
  const pointsGeometry = new THREE.BufferGeometry();
  pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(center, 3));
  const pointsMaterial = new THREE.PointsMaterial();
  const points = new THREE.Points(pointsGeometry, pointsMaterial);
  scene.add(mesh, points);
  scene.updateMatrixWorld(true);

  const instanceMatrix = new THREE.Matrix4();
  mesh.getMatrixAt(0, instanceMatrix);
  const voxelWorld = new THREE.Vector3().applyMatrix4(instanceMatrix).applyMatrix4(mesh.matrixWorld);
  const pointWorld = new THREE.Vector3().fromBufferAttribute(pointsGeometry.getAttribute('position'), 0).applyMatrix4(points.matrixWorld);
  expect(voxelWorld.toArray()).toEqual([-44.75, -34.75, -2.75]);
  expect(pointWorld.toArray()).toEqual(voxelWorld.toArray());
  expect(mesh.matrixWorld.equals(new THREE.Matrix4())).toBe(true);
  expect(points.matrixWorld.equals(new THREE.Matrix4())).toBe(true);

  const camera = new THREE.PerspectiveCamera(42, 1.5, .1, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(70, -90, 70);
  camera.lookAt(0, 0, -1);
  camera.updateMatrixWorld(true);
  expect(camera.up.toArray()).toEqual([0, 0, 1]);
  expect(voxelWorld.clone().project(camera).distanceTo(pointWorld.clone().project(camera))).toBe(0);
  box.dispose(); boxMaterial.dispose(); pointsGeometry.dispose(); pointsMaterial.dispose();
});

test('live renderer uploads all voxel translations and LiDAR positions without an axis conversion', async ({ page, request }) => {
  await apiLogin(request);
  const response = await request.get('/api/runs/infer_mini_cuda_v2');
  expect(response.ok()).toBe(true);
  const detail = await response.json();
  const voxelCount: number = detail.occupancy.point_count;
  const lidarCount: number = detail.lidar.point_count;
  expect(detail.grid_axis_order).toEqual(['x', 'y', 'z']);
  expect(detail.occupancy.fields.slice(0, 3)).toEqual(['x', 'y', 'z']);
  expect(detail.lidar.fields.slice(0, 3)).toEqual(['x', 'y', 'z']);
  expect(detail.lidar.sample_token).toBe(detail.sample_token);

  // Observe the actual typed arrays submitted by the production renderer to
  // WebGL. This hook exists only in this isolated Playwright browser context.
  await page.addInitScript(({ voxelCount, lidarCount }) => {
    const capture: { matrices: Float32Array | null; points: Float32Array | null } = { matrices: null, points: null };
    Object.assign(window, { __voxelCoordinateCapture: capture });
    const record = (data: unknown) => {
      if (!(data instanceof Float32Array)) return;
      if (data.length === voxelCount * 16) capture.matrices = new Float32Array(data);
      if (data.length === lidarCount * 3) capture.points = new Float32Array(data);
    };
    for (const constructor of [window.WebGL2RenderingContext, window.WebGLRenderingContext]) {
      if (!constructor) continue;
      const prototype = constructor.prototype as unknown as Record<string, (...args: unknown[]) => unknown>;
      for (const name of ['bufferData', 'bufferSubData']) {
        const original = prototype[name];
        prototype[name] = function (this: unknown, ...args: unknown[]) {
          record(args[name === 'bufferData' ? 1 : 2]);
          return Reflect.apply(original, this, args);
        };
      }
    }
  }, { voxelCount, lidarCount });

  await pageLogin(page);
  const canvas = page.getByTestId('occupancy-canvas');
  await expect(canvas).toHaveAttribute('data-rendered-instances', String(voxelCount));
  await expect(canvas).toHaveAttribute('data-rendered-points', String(lidarCount));
  const result = await page.evaluate(async ({ occupancyUrl, lidarUrl, voxelCount, lidarCount }) => {
    const capture = (window as unknown as {
      __voxelCoordinateCapture: { matrices: Float32Array | null; points: Float32Array | null };
    }).__voxelCoordinateCapture;
    if (!capture.matrices || !capture.points) throw new Error('Expected actual WebGL instance and point uploads');
    const matrices = capture.matrices, points = capture.points;
    const [occupancy, lidar] = await Promise.all([
      fetch(occupancyUrl).then(response => response.arrayBuffer()),
      fetch(lidarUrl).then(response => response.arrayBuffer()),
    ]);
    const voxels = new DataView(occupancy), cloud = new DataView(lidar);
    let voxelMaxError = 0, pointMaxError = 0, matrixLinearMaxError = 0;
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (let i = 0; i < voxelCount; i++) {
      for (let axis = 0; axis < 3; axis++) {
        voxelMaxError = Math.max(voxelMaxError, Math.abs(matrices[i * 16 + 12 + axis] - voxels.getFloat32((i * 5 + axis) * 4, true)));
      }
      for (let k = 0; k < 16; k++) {
        if (k < 12 || k === 15) matrixLinearMaxError = Math.max(matrixLinearMaxError, Math.abs(matrices[i * 16 + k] - identity[k]));
      }
    }
    for (let i = 0; i < lidarCount; i++) {
      for (let axis = 0; axis < 3; axis++) {
        pointMaxError = Math.max(pointMaxError, Math.abs(points[i * 3 + axis] - cloud.getFloat32((i * 4 + axis) * 4, true)));
      }
    }
    return {
      voxel_count: voxelCount, lidar_point_count: lidarCount,
      voxel_translation_max_abs_error_m: voxelMaxError,
      instance_linear_matrix_max_abs_error: matrixLinearMaxError,
      lidar_position_max_abs_error_m: pointMaxError,
      first_voxel_world_xyz: Array.from(matrices.slice(12, 15)),
      first_lidar_world_xyz: Array.from(points.slice(0, 3)),
    };
  }, { occupancyUrl: detail.occupancy.url, lidarUrl: detail.lidar.url, voxelCount, lidarCount });
  expect(result.voxel_translation_max_abs_error_m).toBe(0);
  expect(result.instance_linear_matrix_max_abs_error).toBe(0);
  expect(result.lidar_position_max_abs_error_m).toBe(0);
  expect(result.first_voxel_world_xyz).toEqual([-49.25, -28.25, 1.75]);
  const evidence = {
    status: 'passed', sample_token: detail.sample_token,
    bounds: detail.bounds, voxel_size: detail.voxel_size, grid_axis_order: detail.grid_axis_order,
    reference_case: { index: [10, 30, 4], center_m: [-44.75, -34.75, -2.75] },
    ...result,
    note: 'Real production WebGL uploads compared with replay API. Native LiDAR XYZ; camera Z up. No inference or artifact modification.',
  };
  fs.writeFileSync(path.join(root, 'work_dirs/web_viewer_coordinate_browser_evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
  console.log('COORDINATE_WEBGL_EVIDENCE', JSON.stringify(evidence));
});
