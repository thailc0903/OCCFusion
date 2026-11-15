const cache = new Map<string, Float32Array>();
let csrfToken = '';

export function rememberCsrf(value: string) { csrfToken = value; }

async function checked(response: Response) {
  if (response.ok) return response;
  const value = await response.json().catch(() => null);
  if (response.status === 401 && typeof window !== 'undefined') {
    cache.clear();
    window.dispatchEvent(new Event('rav25:session-expired'));
  }
  const detail = value?.detail;
  const message = typeof detail === 'string' ? detail : detail?.reason === 'artifact_changed' || detail?.reason === 'approval_does_not_match_artifacts'
    ? 'Artifact đã thay đổi. Approval cũ không áp dụng; cần tạo revision mới và review lại.'
    : detail ? JSON.stringify(detail) : `API trả HTTP ${response.status}`;
  throw new Error(message);
}

export async function getJSON<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await checked(await fetch(url, { cache: 'no-store', credentials: 'same-origin', signal }));
  return response.json();
}

async function writeRequest(url: string, body?: unknown) {
  if (!csrfToken) {
    const session = await getJSON<{ csrf_token: string }>('/api/auth/session');
    rememberCsrf(session.csrf_token);
  }
  return checked(await fetch(url, {
    method: 'POST', credentials: 'same-origin', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: body === undefined ? undefined : JSON.stringify(body),
  }));
}

export async function postJSON<T>(url: string, body?: unknown): Promise<T> {
  return (await writeRequest(url, body)).json();
}

export async function downloadApproved(runId: string) {
  return downloadCollectionOrRun(`/api/runs/${encodeURIComponent(runId)}/export`, `${runId}-approved.zip`);
}

export async function downloadCollection(collectionId: string) {
  return downloadCollectionOrRun(`/api/collections/${encodeURIComponent(collectionId)}/export`, `${collectionId}-collection-approved.zip`);
}

export async function downloadCamera(endpoint: string) {
  return downloadCollectionOrRun(endpoint + '/export', 'camera-frame-approved.zip');
}

async function downloadCollectionOrRun(endpoint: string, fallback: string) {
  const response = await writeRequest(endpoint);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = response.headers.get('content-disposition')?.match(/filename="([^"]+)"/)?.[1] || fallback;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function getPacked(url: string, count: number, stride: number, signal?: AbortSignal) {
  if (cache.has(url)) return cache.get(url)!;
  const response = await checked(await fetch(url, { cache: 'no-store', credentials: 'same-origin', signal }));
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== count * stride * 4) throw new Error('Kích thước payload không khớp schema API.');
  const view = new DataView(buffer);
  const values = new Float32Array(count * stride);
  for (let i = 0; i < values.length; i++) {
    values[i] = view.getFloat32(i * 4, true);
    if (!Number.isFinite(values[i])) throw new Error('Payload có giá trị không hữu hạn.');
  }
  cache.set(url, values);
  // Bound browser-side replay cache as well as backend LRU cache.
  if (cache.size > 16) cache.delete(cache.keys().next().value!);
  return values;
}

export function clearPayloadCache() { cache.clear(); }
