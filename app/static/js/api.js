/**
 * FaceMakeIt API Client
 */
const API = {
    _token: localStorage.getItem('api_key') || '',

    setToken(key) {
        this._token = key;
        localStorage.setItem('api_key', key);
    },

    _headers() {
        const h = {};
        if (this._token) h['Authorization'] = `Bearer ${this._token}`;
        return h;
    },

    async _handle(res) {
        const body = await res.json().catch(() => ({}));
        if (!res.ok || body.success === false) {
            throw new Error(body.error || `Request failed (${res.status})`);
        }
        return body.data;
    },

    async upload(file) {
        const form = new FormData();
        form.append('file', file);
        const res = await fetch('/api/upload', { method: 'POST', body: form, headers: this._headers() });
        return this._handle(res);
    },

    async process(taskId) {
        const res = await fetch(`/api/process/${taskId}`, { method: 'POST', headers: this._headers() });
        return this._handle(res);
    },

    async status(taskId) {
        const res = await fetch(`/api/status/${taskId}`, { headers: this._headers() });
        return this._handle(res);
    },

    async faces(taskId) {
        const res = await fetch(`/api/faces/${taskId}`, { headers: this._headers() });
        return this._handle(res);
    },

    resultUrl: (taskId) => `/api/result/${taskId}`,
    originalUrl: (taskId, ext) => `/uploads/${taskId}/original${ext}`,
};
