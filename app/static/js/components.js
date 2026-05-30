/**
 * FaceMakeIt UI Helpers
 */
const UI = {
    showView(viewId) {
        document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
        document.getElementById(viewId).classList.remove('hidden');
    },

    updateProgress(stage, pct) {
        const labels = {
            starting: 'Memulai...',
            detecting: 'Mendeteksi wajah...',
            scoring: 'Memilih subjek utama...',
            masking: 'Membuat area penghapusan...',
            inpainting: 'Menghapus wajah background...',
            blurring: 'Blur wajah background...',
            done: 'Selesai!',
            no_faces: 'Tidak ada wajah terdeteksi',
            no_background: 'Hanya 1 wajah, tidak perlu dihapus',
            error: 'Terjadi kesalahan',
        };
        document.getElementById('stage-label').textContent = labels[stage] || stage;
        document.getElementById('progress-bar').style.width = `${pct}%`;
        document.getElementById('progress-pct').textContent = `${pct}%`;
    },

    showFaceInfo(detected, removed) {
        const el = document.getElementById('face-info');
        el.classList.remove('hidden');
        document.getElementById('badge-detected').textContent = `${detected} wajah terdeteksi`;
        document.getElementById('badge-removed').textContent = `${removed} akan dihapus`;
    },

    showResult(taskId, ext, faces) {
        document.getElementById('result-original').src = API.originalUrl(taskId, ext);
        document.getElementById('result-output').src = API.resultUrl(taskId);

        const removed = faces.filter(f => !f.is_main).length;
        document.getElementById('result-summary').textContent =
            `${faces.length} wajah terdeteksi, ${removed} dihapus.`;

        if (faces.length > 0) {
            document.getElementById('face-details').classList.remove('hidden');
            document.getElementById('face-list').innerHTML = faces.map((f, i) => `
                <div class="flex items-center justify-between py-2 ${i > 0 ? 'border-t border-border' : ''}">
                    <div class="flex items-center gap-2">
                        <span class="badge ${f.is_main ? 'badge-main' : 'badge-remove'}">
                            ${f.is_main ? 'Subjek Utama' : 'Background'}
                        </span>
                        <span class="text-sm">Wajah #${i + 1}</span>
                    </div>
                    <div class="text-xs text-muted-foreground">
                        Score: ${f.score.toFixed(3)} &middot; Conf: ${(f.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            `).join('');
        }

        document.getElementById('btn-download').href = API.resultUrl(taskId);
    },

    showError(message) {
        document.getElementById('error-message').textContent = message;
        UI.showView('view-error');
    },

    formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },
};
