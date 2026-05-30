/**
 * FaceMakeIt - Main Application
 */
(function () {
    'use strict';

    let selectedFile = null;
    let currentTaskId = null;
    let pollInterval = null;

    const $ = (id) => document.getElementById(id);

    // DOM refs
    const dropzone = $('dropzone');
    const fileInput = $('file-input');
    const btnProcess = $('btn-process');
    const btnReset = $('btn-reset');

    // --- Upload ---
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

    function handleFile(file) {
        if (!file.type.match(/^image\/(jpeg|png|webp)$/)) return alert('File harus JPG, PNG, atau WebP');
        if (file.size > 20 * 1024 * 1024) return alert('File terlalu besar (maks 20MB)');

        selectedFile = file;
        $('preview-img').src = URL.createObjectURL(file);
        $('preview-name').textContent = file.name;
        $('preview-size').textContent = UI.formatSize(file.size);

        $('dropzone-content').classList.add('hidden');
        $('dropzone-preview').classList.remove('hidden');
        btnProcess.disabled = false;
        btnReset.classList.remove('hidden');
    }

    btnReset.addEventListener('click', (e) => { e.stopPropagation(); resetUpload(); });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        $('dropzone-content').classList.remove('hidden');
        $('dropzone-preview').classList.add('hidden');
        btnProcess.disabled = true;
        btnProcess.textContent = 'Proses Foto';
        btnReset.classList.add('hidden');
    }

    // --- Mode selector ---
    let selectedMode = 'remove';
    document.querySelectorAll('.mode-option').forEach(opt => {
        opt.addEventListener('click', () => {
            document.querySelectorAll('.mode-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            opt.querySelector('input').checked = true;
            selectedMode = opt.dataset.mode;
        });
    });

    // --- Process ---
    btnProcess.addEventListener('click', async () => {
        if (!selectedFile) return;

        try {
            btnProcess.disabled = true;
            btnProcess.textContent = 'Mengupload...';

            const { task_id } = await API.upload(selectedFile);
            currentTaskId = task_id;

            UI.showView('view-processing');
            UI.updateProgress('starting', 5);
            await API.process(task_id, selectedMode);
            startPolling(task_id);
        } catch (err) {
            UI.showError(err.message);
            btnProcess.disabled = false;
            btnProcess.textContent = 'Proses Foto';
        }
    });

    // --- Polling ---
    function startPolling(taskId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const s = await API.status(taskId);
                UI.updateProgress(s.stage, s.progress);

                if (s.faces_detected !== undefined) {
                    UI.showFaceInfo(s.faces_detected, s.faces_removed || 0);
                }

                if (s.status === 'done') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    await showResult(taskId);
                } else if (s.status === 'error') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    UI.showError(s.error || s.message || 'Processing failed');
                }
            } catch {
                clearInterval(pollInterval);
                pollInterval = null;
                UI.showError('Connection lost');
            }
        }, 500);
    }

    async function showResult(taskId) {
        try {
            const [status, faces] = await Promise.all([
                API.status(taskId),
                API.faces(taskId).catch(() => []),
            ]);

            UI.showView('view-result');
            UI.showResult(taskId, status.extension || '.jpg', faces);
        } catch (err) {
            UI.showError(err.message);
        }
    }

    // --- Result actions ---
    $('btn-another').addEventListener('click', () => { resetUpload(); UI.showView('view-upload'); });
    $('btn-retry').addEventListener('click', () => { resetUpload(); UI.showView('view-upload'); });

    // --- API Key ---
    const keyInput = $('api-key-input');
    if (API._token) keyInput.value = API._token;
    $('btn-save-key').addEventListener('click', () => {
        API.setToken(keyInput.value.trim());
    });

    // --- Init ---
    UI.showView('view-upload');
})();
