
            let currentProcessedImageUrl = null;
            let currentProcessedFilename = null;
            let currentMasterImageUrl = null;
        
            document.addEventListener("DOMContentLoaded", function() {
                try {
                    const canvas = document.createElement('canvas');
                    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                    let hasHighEndGPU = false;
                    let gpuName = "Generic CPU/Basic GPU";
                
                    if (gl) {
                        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                        if (debugInfo) {
                            gpuName = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL).toLowerCase();
                            if (gpuName.includes('nvidia') || gpuName.includes('amd') || gpuName.includes('radeon') || gpuName.includes('apple') || gpuName.includes('rtx')) {
                                hasHighEndGPU = true;
                            }
                        }
                    }
                
                    const procModeEl = document.getElementById('procmode-selected-text');
                    const liveGpuEl = document.getElementById('live-gpu-name');
                    const liveCpuEl = document.getElementById('live-cpu-status');
                
                    const savedUrl = localStorage.getItem('nexus_colab_url');
                
                    if (hasHighEndGPU) {
                        if (!savedUrl) {
                            procModeEl.innerText = "GPU Accelerated (Fast)";
                        } else {
                            procModeEl.innerText = "External Cloud GPU";
                        }
                        document.getElementById('procmode-menu').querySelector('li:nth-child(1)').className = "px-4 py-2.5 text-sm font-bold text-blue-600 bg-blue-50 cursor-pointer transition-colors";
                        if(liveGpuEl) liveGpuEl.innerText = gpuName.split(' ')[0] + " Active";
                        if(liveCpuEl) liveCpuEl.innerText = "Optimized";
                    } else {
                        if (!savedUrl) {
                            procModeEl.innerText = "CPU Precision (Slow)";
                        } else {
                            procModeEl.innerText = "External Cloud GPU";
                        }
                        document.getElementById('procmode-menu').querySelector('li:nth-child(2)').className = "px-4 py-2.5 text-sm font-bold text-blue-600 bg-blue-50 cursor-pointer transition-colors";
                        if(liveGpuEl) liveGpuEl.innerText = "Basic GPU / Off";
                        if(liveCpuEl) liveCpuEl.innerText = "CPU Handling";
                    }
                } catch (e) {
                    document.getElementById('procmode-selected-text').innerText = "GPU Accelerated (Fast)";
                    document.getElementById('live-gpu-name').innerText = "Hardware Active";
                }
            });
        
            function switchTab(tabName, btn) {
                document.querySelectorAll('.setting-tab').forEach(b => {
                    b.classList.remove('active', 'text-blue-600', 'border-blue-600');
                    b.classList.add('text-slate-500', 'border-transparent');
                });
                btn.classList.add('active', 'text-blue-600', 'border-blue-600');
                btn.classList.remove('text-slate-500', 'border-transparent');
            
                const allTabs = ['tab-primary', 'tab-advanced', 'tab-face', 'tab-export', 'tab-server', 'tab-batch'];
                allTabs.forEach(t => {
                    const el = document.getElementById(t);
                    if(el) {
                        el.classList.add('hidden');
                        if (t === 'tab-batch') el.classList.remove('flex');
                        else el.classList.remove('grid');
                    }
                });
            
                const target = document.getElementById('tab-' + tabName);
                if(target) {
                    target.classList.remove('hidden');
                    if (tabName === 'batch') target.classList.add('flex');
                    else target.classList.add('grid');
                }
                
                // Toggle canvases
                const mainCanvas = document.getElementById('canvas-card');
                const batchCanvas = document.getElementById('batch-canvas-card');
                
                if (tabName === 'batch') {
                    if(mainCanvas) { mainCanvas.classList.add('hidden'); mainCanvas.classList.remove('flex'); }
                    if(batchCanvas) { batchCanvas.classList.remove('hidden'); batchCanvas.classList.add('flex'); }
                } else {
                    if(mainCanvas) { mainCanvas.classList.remove('hidden'); mainCanvas.classList.add('flex'); }
                    if(batchCanvas) { batchCanvas.classList.add('hidden'); batchCanvas.classList.remove('flex'); }
                }
            }
        
            // Modal Button Redirect Logic
            function goToExportTab() {
                closeProgressModal();
                const exportTabBtn = document.getElementById('export-tab-btn');
                if (exportTabBtn) {
                    switchTab('export', exportTabBtn);
                }
            }
            
            function resetApp() {
                closeProgressModal();
                document.getElementById('preview-state').classList.add('hidden');
                document.getElementById('preview-state').classList.remove('flex');
                document.getElementById('bottom-action-bar').classList.add('hidden');
                document.getElementById('bottom-action-bar').classList.remove('flex');
                
                document.getElementById('upload-state').classList.remove('hidden');
                document.getElementById('upload-state').classList.add('flex');
                
                document.getElementById('file-input').value = "";
                window.currentSingleFile = null;
                currentProcessedImageUrl = null;
                currentMasterImageUrl = null;
            }
            
            function resetBatch() {
                batchFiles = [];
                batchResults = [];
                document.getElementById('batch-upload-state').classList.remove('hidden');
                document.getElementById('batch-file-list').classList.add('hidden');
                document.getElementById('batch-file-list').innerHTML = '';
                document.getElementById('batch-file-input').value = "";
                
                const btn = document.getElementById('batch-start-btn');
                if(btn) {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="ph-bold ph-lightning"></i> Run Batch Upscaler';
                    btn.classList.add('hidden');
                }
                const dlBtn = document.getElementById('batch-download-btn');
                if(dlBtn) dlBtn.classList.add('hidden');
                
                const addMoreBtn = document.getElementById('batch-add-more-btn');
                if(addMoreBtn) addMoreBtn.classList.add('hidden');
                
                document.getElementById('batch-status-text').innerText = "0 files loaded";
                document.getElementById('batch-progress-fill').style.width = '0%';
            }
        
            let currentZoom = 100;
            const zoomContainer = document.getElementById('zoom-container');
        
            function updateZoomUI() {
                document.getElementById('zoom-text').innerText = Math.round(currentZoom) + '%';
                zoomContainer.style.transform = `scale(${currentZoom / 100})`;
            }
        
            function changeZoom(amount) {
                currentZoom += amount;
                if (currentZoom < 10) currentZoom = 10; 
                updateZoomUI();
            }
        
            document.getElementById('preview-state').addEventListener('wheel', (e) => {
                const isZoomEnabled = document.getElementById('toggle-scroll-zoom').checked;
                if (!isZoomEnabled) return; 

                e.preventDefault(); 
                const delta = e.deltaY > 0 ? -10 : 10; 
                currentZoom += delta;
                if (currentZoom < 10) currentZoom = 10;
                updateZoomUI();
            });
        
            function updateVal(slider) {
                const span = slider.previousElementSibling.querySelector('.val');
                if(span) span.innerText = slider.value + '%';
            
                const strength = document.getElementById('slider-strength').value;
                const texture = document.getElementById('slider-texture').value;
                const contrast = 100 + (strength - 50) * 0.4;
                const saturate = 100 + (texture - 50) * 0.5;

                const upImg = document.getElementById('upscaled-img');
                if(upImg) upImg.style.filter = `contrast(${contrast}%) saturate(${saturate}%)`;
            }
        
            const dropZone = document.getElementById('drop-zone');
            const uploadState = document.getElementById('upload-state');
            const previewState = document.getElementById('preview-state');
            const bottomBar = document.getElementById('bottom-action-bar');
            const origImg = document.getElementById('original-img');
            const upImg = document.getElementById('upscaled-img');

            let batchFiles = [];
            let batchResults = [];
            let batchCurrentIndex = 0;

            function handleUpload(files) {
                if(!files || files.length === 0) return;
                
                const file = files[0]; // Always process the first file in single mode
                const url = URL.createObjectURL(file);
            
                origImg.src = url;
                upImg.src = url; 
            
                const img = new Image();
                img.onload = function() {
                    document.getElementById('res-in').innerText = this.width + "x" + this.height;
                    document.getElementById('image-wrapper').style.width = '100%';
                };
                img.src = url;
            
                document.getElementById('file-name-display').innerText = file.name;
                document.getElementById('bottom-file-name').innerText = file.name;

                uploadState.classList.add('hidden');
                uploadState.classList.remove('flex');
                previewState.classList.remove('hidden');
                previewState.classList.add('flex');
                bottomBar.classList.remove('hidden');
                bottomBar.classList.add('flex');

                currentZoom = 100;
                updateZoomUI();
                setTimeout(syncSplit, 50);
                
                window.currentSingleFile = file;
            }
            
            function handleBatchUpload(files) {
                if(!files || files.length === 0) return;
                
                const newFiles = Array.from(files);
                batchFiles = batchFiles.concat(newFiles);
                batchResults = batchResults.concat(new Array(newFiles.length).fill(null));
                
                document.getElementById('batch-upload-state').classList.add('hidden');
                document.getElementById('batch-file-list').classList.remove('hidden');
                document.getElementById('batch-start-btn').classList.remove('hidden');
                document.getElementById('batch-add-more-btn').classList.remove('hidden');
                
                renderBatchUI();
            }
        
            function renderBatchUI() {
                const list = document.getElementById('batch-file-list');
                list.innerHTML = '';
                document.getElementById('batch-status-text').innerText = `${batchFiles.length} files ready to process`;
                document.getElementById('batch-progress-fill').style.width = '0%';
                
                const template = document.getElementById('batch-card-template');
                batchFiles.forEach((f, idx) => {
                    const objUrl = URL.createObjectURL(f);
                    const clone = template.content.cloneNode(true);
                    
                    const card = clone.firstElementChild;
                    card.onclick = () => openLightbox(idx);
                    
                    clone.querySelector('.batch-img').src = objUrl;
                    clone.querySelector('.batch-name').innerText = f.name;
                    
                    clone.querySelector('.batch-overlay').id = `batch-overlay-${idx}`;
                    clone.querySelector('.batch-prog').id = `batch-prog-${idx}`;
                    clone.querySelector('.batch-log').id = `batch-log-${idx}`;
                    
                    list.appendChild(clone);
                });
            }

            let isLightboxZoomed = false;

            function updateZoomClasses() {
                const upImg = document.getElementById('lightbox-upscaled-img');
                const origImg = document.getElementById('lightbox-original-img');
                const singleImg = document.getElementById('lightbox-single-img');
                const icon = document.getElementById('lightbox-zoom-icon');
                const sliderWrapper = document.getElementById('lightbox-slider-wrapper');
                const singleWrapper = document.getElementById('lightbox-single-wrapper');
                
                if (isLightboxZoomed) {
                    upImg.className = "w-auto h-auto max-w-none max-h-none pointer-events-none block";
                    origImg.className = "absolute top-0 left-0 h-full w-auto max-w-none pointer-events-none";
                    singleImg.className = "w-auto h-auto max-w-none max-h-none pointer-events-none block";
                    if(icon) icon.className = "ph-bold ph-magnifying-glass-minus text-xl";
                    sliderWrapper.classList.remove('items-center', 'justify-center');
                    sliderWrapper.classList.add('items-start', 'justify-start');
                    singleWrapper.classList.remove('items-center', 'justify-center');
                    singleWrapper.classList.add('items-start', 'justify-start');
                } else {
                    upImg.className = "max-w-full max-h-screen object-contain pointer-events-none block";
                    origImg.className = "absolute top-0 left-0 h-full w-full max-w-none object-contain pointer-events-none";
                    singleImg.className = "max-w-full max-h-screen object-contain pointer-events-none block";
                    if(icon) icon.className = "ph-bold ph-magnifying-glass-plus text-xl";
                    sliderWrapper.classList.add('items-center', 'justify-center');
                    sliderWrapper.classList.remove('items-start', 'justify-start');
                    singleWrapper.classList.add('items-center', 'justify-center');
                    singleWrapper.classList.remove('items-start', 'justify-start');
                }
            }

            function toggleLightboxZoom(e) {
                if(e) e.stopPropagation();
                isLightboxZoomed = !isLightboxZoomed;
                updateZoomClasses();
                syncLightboxSplit();
            }

            function openLightbox(idx) {
                const modal = document.getElementById('lightbox-modal');
                const singleWrapper = document.getElementById('lightbox-single-wrapper');
                const sliderWrapper = document.getElementById('lightbox-slider-wrapper');
                const title = document.getElementById('lightbox-title');
                const zoomBtn = document.getElementById('lightbox-zoom-btn');
                
                const file = batchFiles[idx];
                const res = batchResults[idx];
                const origUrl = URL.createObjectURL(file);

                isLightboxZoomed = false;
                updateZoomClasses();

                if (res && res.status === 'success' && res.processed_path) {
                    singleWrapper.classList.add('hidden');
                    singleWrapper.classList.remove('flex');
                    sliderWrapper.classList.remove('hidden');
                    sliderWrapper.classList.add('flex');
                    zoomBtn.classList.remove('hidden');
                    
                    document.getElementById('lightbox-upscaled-img').src = res.processed_path;
                    document.getElementById('lightbox-original-img').src = origUrl;
                    
                    title.innerHTML = `<span class="text-emerald-400"><i class="ph-bold ph-check-circle"></i> Upscaled:</span> ${res.filename || file.name}`;
                    
                    setTimeout(() => {
                        syncLightboxSplit();
                    }, 50);
                } else {
                    sliderWrapper.classList.add('hidden');
                    sliderWrapper.classList.remove('flex');
                    singleWrapper.classList.remove('hidden');
                    singleWrapper.classList.add('flex');
                    zoomBtn.classList.remove('hidden');
                    
                    document.getElementById('lightbox-single-img').src = origUrl;
                    title.innerHTML = `<span class="text-blue-400"><i class="ph-bold ph-image"></i> Original:</span> ${file.name}`;
                }
                
                modal.classList.remove('hidden');
                modal.classList.add('flex');
                setTimeout(() => modal.classList.remove('opacity-0'), 10);
            }

            const lbSlider = document.getElementById('lightbox-slider');
            const lbClip = document.getElementById('lightbox-compare-clip');
            const lbHandle = document.getElementById('lightbox-compare-handle');
            const lbOrig = document.getElementById('lightbox-original-img');
            
            function syncLightboxSplit() {
                if(!lbSlider) return;
                const val = lbSlider.value;
                lbClip.style.width = val + '%';
                lbHandle.style.left = val + '%';
                const upImg = document.getElementById('lightbox-upscaled-img');
                if(upImg) {
                    lbOrig.style.width = upImg.offsetWidth + 'px';
                }
            }
            if(lbSlider) lbSlider.addEventListener('input', syncLightboxSplit);
            window.addEventListener('resize', syncLightboxSplit);

            function openLightbox(idx) {
                const f = batchFiles[idx];
                const r = batchResults[idx];
                
                document.getElementById('lightbox-title').innerText = f.name;
                const origUrl = URL.createObjectURL(f);
                
                const upImg = document.getElementById('lightbox-upscaled-img');
                const origImg = document.getElementById('lightbox-original-img');
                const sliderWrapper = document.getElementById('lightbox-slider-wrapper');
                
                origImg.src = origUrl;
                
                if (r && r.status === 'success') {
                    upImg.src = r.processed_path + "?t=" + new Date().getTime();
                    document.getElementById('lightbox-slider').value = 50;
                    syncLightboxSplit();
                } else {
                    upImg.src = origUrl;
                    document.getElementById('lightbox-slider').value = 0;
                    syncLightboxSplit();
                }

                // Reset zoom state
                isLightboxZoomed = false;
                updateZoomClasses();

                const previewState = document.getElementById('batch-preview-state');
                const fileList = document.getElementById('batch-file-list');
                
                fileList.classList.add('hidden');
                previewState.classList.remove('hidden');
                previewState.classList.add('flex');
            }

            function closeLightbox() {
                const previewState = document.getElementById('batch-preview-state');
                const fileList = document.getElementById('batch-file-list');
                
                previewState.classList.add('hidden');
                previewState.classList.remove('flex');
                fileList.classList.remove('hidden');
            }

            function setBatchView(mode, btn) {
                const parent = btn.closest('.flex');
                parent.querySelectorAll('.view-btn').forEach(b => {
                    b.classList.remove('active', 'text-blue-400', 'bg-slate-900', 'shadow-sm');
                    b.classList.add('text-slate-400');
                });
                btn.classList.add('active', 'text-blue-400', 'bg-slate-900', 'shadow-sm');
                btn.classList.remove('text-slate-400');
                
                const splitWrapper = document.getElementById('lightbox-slider-wrapper');
                const singleWrapper = document.getElementById('lightbox-single-wrapper');
                
                if(mode === 'split') {
                    splitWrapper.classList.remove('hidden');
                    splitWrapper.classList.add('flex');
                    singleWrapper.classList.add('hidden');
                    singleWrapper.classList.remove('flex');
                } else {
                    splitWrapper.classList.add('hidden');
                    splitWrapper.classList.remove('flex');
                    singleWrapper.classList.remove('hidden');
                    singleWrapper.classList.add('flex');
                }
            }

            dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.border = '2px dashed #3b82f6'; });
            dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.style.border = 'none'; });
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.style.border = 'none';
                if(e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files);
            });
            
            const batchCanvasDrop = document.getElementById('batch-canvas-card');
            batchCanvasDrop.addEventListener('dragover', (e) => { e.preventDefault(); batchCanvasDrop.style.border = '2px dashed #3b82f6'; });
            batchCanvasDrop.addEventListener('dragleave', (e) => { e.preventDefault(); batchCanvasDrop.style.border = '1px solid #e2e8f0'; });
            batchCanvasDrop.addEventListener('drop', (e) => {
                e.preventDefault();
                batchCanvasDrop.style.border = '1px solid #e2e8f0';
                if(e.dataTransfer.files.length > 0) handleBatchUpload(e.dataTransfer.files);
            });
        
            const slider = document.getElementById('compare-slider');
            const clip = document.getElementById('compare-clip');
            const handle = document.getElementById('compare-handle');
            const wrapper = document.getElementById('image-wrapper');
        
            function syncSplit() {
                const val = slider.value;
                clip.style.width = val + '%';
                handle.style.left = val + '%';
                if(wrapper) origImg.style.width = wrapper.offsetWidth + 'px';
            }
            slider.addEventListener('input', syncSplit);
            window.addEventListener('resize', syncSplit);
        
            function setView(mode, btn) {
                document.querySelectorAll('.view-btn').forEach(b => {
                    b.classList.remove('active', 'text-blue-600', 'bg-white', 'shadow-sm');
                    b.classList.add('text-slate-500');
                });
                btn.classList.add('active', 'text-blue-600', 'bg-white', 'shadow-sm');
                btn.classList.remove('text-slate-500');
            
                if(mode === 'single') {
                    clip.style.width = '0%';
                    handle.style.display = 'none';
                    slider.style.display = 'none';
                } else {
                    clip.style.width = slider.value + '%';
                    handle.style.display = 'flex';
                    slider.style.display = 'block';
                }
            }
        
            function toggleDropdown(menuId, event) {
                event.stopPropagation();
                const menu = document.getElementById(menuId);
                const isHidden = menu.classList.contains('hidden');
                document.querySelectorAll('[id$="-menu"]').forEach(m => m.classList.add('hidden'));
                if (isHidden) menu.classList.remove('hidden');
            }
        
            function selectOption(dropdownName, value, element) {
                const selectedText = document.getElementById(`${dropdownName}-selected-text`);
                if(selectedText) selectedText.textContent = value;

                const menu = document.getElementById(`${dropdownName}-menu`);
                if(menu) {
                    menu.querySelectorAll('li').forEach(item => {
                        if(!item.classList.contains('pointer-events-none')) {
                            item.className = "px-4 py-2.5 text-sm font-bold text-slate-600 hover:bg-blue-50 hover:text-blue-600 cursor-pointer transition-colors";
                        }
                    });
                    element.className = "px-4 py-2.5 text-sm font-bold text-blue-600 bg-blue-50 cursor-pointer transition-colors";
                    menu.classList.add('hidden');
                }
            }
            window.addEventListener('click', () => document.querySelectorAll('[id$="-menu"]').forEach(m => m.classList.add('hidden')));
        
            function downloadUpscaledImage() {
                if (!currentMasterImageUrl) {
                    alert("Please wait for the process to finish.");
                    return;
                }
            
                const btn = document.querySelector('button[onclick="downloadUpscaledImage()"]');
                const originalHtml = btn.innerHTML;
                btn.innerHTML = `<i class="ph-bold ph-spinner animate-spin"></i> Downloading...`;
                btn.disabled = true;

                setTimeout(() => {
                    try {
                        const link = document.createElement('a');
                        // Fix for DNS error: window.location.origin forces the correct server address
                        link.href = window.location.origin + currentMasterImageUrl; 
                        link.download = currentProcessedFilename;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    } catch (error) {
                        alert("Download Failed.");
                    } finally {
                        btn.innerHTML = originalHtml;
                        btn.disabled = false;
                    }
                }, 500);
            }
        
            const progressModal = document.getElementById('progress-modal');
            const modalBox = document.getElementById('progress-modal-box');
            const terminal = document.getElementById('terminal-steps');

            async function startBatchProcessing() {
                const mode = document.getElementById('procmode-selected-text').innerText;
                if (mode.includes('Cloud')) {
                    const colabUrl = localStorage.getItem('nexus_colab_url');
                    if (!colabUrl) {
                        alert("Please connect a Cloud GPU url first from the top right button.");
                        return;
                    }
                }

                const btn = document.getElementById('batch-start-btn');
                btn.disabled = true;
                btn.innerHTML = '<i class="ph-bold ph-spinner animate-spin"></i> Processing...';
                
                for (let i = 0; i < batchFiles.length; i++) {
                    await processSingleBatchImage(i);
                    const totalProg = Math.round(((i + 1) / batchFiles.length) * 100);
                    document.getElementById('batch-progress-fill').style.width = totalProg + '%';
                    document.getElementById('batch-status-text').innerText = `Processed ${i + 1} of ${batchFiles.length}`;
                }
                
                btn.classList.add('hidden');
                document.getElementById('batch-download-btn').classList.remove('hidden');
                document.getElementById('batch-status-text').innerText = "All Complete! Ready for export.";
            }

            async function processSingleBatchImage(idx) {
                const file = batchFiles[idx];
                const overlay = document.getElementById(`batch-overlay-${idx}`);
                const progBar = document.getElementById(`batch-prog-${idx}`);
                
                overlay.classList.remove('hidden');
                
                const formData = new FormData();
                formData.append('image', file);
                const taskId = "batch_" + Math.random().toString(36).substring(2, 9);
                formData.append('task_id', taskId);
                
                // Get current settings (using same for all batch items as planned)
                formData.append('factor', document.getElementById('upscale-selected-text').innerText);
                formData.append('model', document.getElementById('aimodel-selected-text').innerText);
                
                let mode = document.getElementById('procmode-selected-text').innerText;
                if (mode.includes('Cloud')) mode = 'External Cloud GPU';
                formData.append('mode', mode);
                
                const strength = document.getElementById('slider-strength').value;
                const noise = document.getElementById('slider-noise').value;
                const texture = document.getElementById('slider-texture').value;
                const artifact = document.getElementById('slider-artifact').value;
                formData.append('settings', JSON.stringify({ strength: strength, noise: noise, texture: texture, artifact: artifact }));
            
                const faceEnable = document.getElementById('check-face-restore').checked;
                const faceIdentity = document.getElementById('slider-identity').value;
                const faceSkin = document.getElementById('slider-skin').value;
                formData.append('face', JSON.stringify({ enabled: faceEnable, identity: faceIdentity, skin_tone: faceSkin }));
            
                const exportFormat = document.getElementById('format-selected-text').innerText;
                const exportColor = document.getElementById('color-selected-text').innerText;
                const exportDPI = document.getElementById('dpi-selected-text').innerText;
                formData.append('export', JSON.stringify({ format: exportFormat, color_space: exportColor, dpi: exportDPI }));
                
                let colabUrl = localStorage.getItem('nexus_colab_url');
                if(colabUrl && colabUrl.trim() !== "") {
                    formData.append('colab_url', colabUrl.trim());
                }

                const checkInterval = setInterval(async () => {
                    try {
                        const res = await fetch(`/api/progress?task_id=${taskId}`);
                        const data = await res.json();
                        if (data.percent !== undefined) progBar.style.width = data.percent + '%';
                        if (data.log) document.getElementById(`batch-log-${idx}`).innerText = data.log;
                    } catch(e) {}
                }, 1000);
                
                try {
                    const response = await fetch('/api/process-upscale', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    clearInterval(checkInterval);
                    overlay.classList.add('hidden');
                    
                    if (result.status === 'success') {
                        progBar.style.width = '100%';
                        progBar.classList.replace('bg-blue-500', 'bg-emerald-500');
                        document.getElementById(`batch-log-${idx}`).innerText = "Success! Image Upscaled.";
                        batchResults[idx] = result;
                    } else {
                        progBar.classList.replace('bg-blue-500', 'bg-red-500');
                        document.getElementById(`batch-log-${idx}`).innerText = "Error: " + (result.message || "Failed");
                        document.getElementById(`batch-log-${idx}`).classList.replace('text-slate-400', 'text-red-500');
                    }
                } catch(e) {
                    clearInterval(checkInterval);
                    overlay.classList.add('hidden');
                    progBar.classList.replace('bg-blue-500', 'bg-red-500');
                    document.getElementById(`batch-log-${idx}`).innerText = "Error: Network Timeout / Tunnel Dead";
                    document.getElementById(`batch-log-${idx}`).classList.replace('text-slate-400', 'text-red-500');
                }
            }

            async function downloadAllBatch() {
                const dlBtn = document.getElementById('batch-download-btn');
                const origHtml = dlBtn.innerHTML;
                dlBtn.innerHTML = '<i class="ph-bold ph-spinner animate-spin"></i> Zipping...';
                dlBtn.disabled = true;
                
                try {
                    const zip = new JSZip();
                    let hasFiles = false;
                    for (let i = 0; i < batchResults.length; i++) {
                        const res = batchResults[i];
                        if(res && res.processed_path) {
                            try {
                                const response = await fetch(window.location.origin + res.processed_path);
                                const blob = await response.blob();
                                zip.file(res.filename || `upscaled_${i}.png`, blob);
                                hasFiles = true;
                            } catch(e) {
                                console.error("Error fetching file for zip:", e);
                            }
                        }
                    }
                    if (hasFiles) {
                        dlBtn.innerHTML = '<i class="ph-bold ph-spinner animate-spin"></i> Generating...';
                        const content = await zip.generateAsync({type:"blob"});
                        const link = document.createElement('a');
                        link.href = URL.createObjectURL(content);
                        link.download = "Nexus_Batch_Upscaled.zip";
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        URL.revokeObjectURL(link.href);
                    }
                } catch(err) {
                    console.error("Zip generation error:", err);
                }
                
                dlBtn.innerHTML = origHtml;
                dlBtn.disabled = false;
            }


            async function processImage(event) {
                if (event) event.preventDefault();
                
                // Batch mode check removed to fix ReferenceError
                        
                if (uploadState.classList.contains('flex') || !uploadState.classList.contains('hidden')) {
                    alert("Please drag and drop an image first.");
                    return;
                }
            
                const fileInput = document.getElementById('file-input');
                let file = window.currentSingleFile;
                if (!file) {
                    if (fileInput.files.length > 0) file = fileInput.files[0];
                    else { alert("Please drop an image first."); return; }
                }
            
                const factor = document.getElementById('upscale-selected-text').innerText;
                const model = document.getElementById('aimodel-selected-text').innerText;
                let mode = document.getElementById('procmode-selected-text').innerText;
                if (mode.includes('Cloud')) {
                    mode = 'External Cloud GPU';
                }
            
                const strength = document.getElementById('slider-strength').value;
                const noise = document.getElementById('slider-noise').value;
                const texture = document.getElementById('slider-texture').value;
                const artifact = document.getElementById('slider-artifact').value;
            
                const faceEnable = document.getElementById('check-face-restore').checked;
                const faceIdentity = document.getElementById('slider-identity').value;
                const faceSkin = document.getElementById('slider-skin').value;
            
                const exportFormat = document.getElementById('format-selected-text').innerText;
                const exportColor = document.getElementById('color-selected-text').innerText;
                const exportDPI = document.getElementById('dpi-selected-text').innerText;
            
                // 🚀 NAYA: Generate Unique Task ID for Real-Time Tracking
                const taskId = "task_" + Math.random().toString(36).substr(2, 9);
            
                const formData = new FormData();
                formData.append('image', file);
                formData.append('task_id', taskId); // Task ID backend ko bhej rahe hain
                formData.append('factor', factor);
                formData.append('model', model);
                formData.append('mode', mode);
                formData.append('settings', JSON.stringify({ strength: strength, noise: noise, texture: texture, artifact: artifact }));
                formData.append('face', JSON.stringify({ enabled: faceEnable, identity: faceIdentity, skin_tone: faceSkin }));
                formData.append('export', JSON.stringify({ format: exportFormat, color_space: exportColor, dpi: exportDPI }));
                
                if (mode === 'External Cloud GPU') {
                    const colabUrl = localStorage.getItem('nexus_colab_url');
                    if (!colabUrl) {
                        alert("Please connect a Cloud GPU url first from the top right button.");
                        return;
                    }
                    formData.append('colab_url', colabUrl);
                }
            

                
                document.getElementById('modal-model-name').innerText = model;
                document.getElementById('modal-factor-name').innerText = factor;
            
                // Reset Modal Animations
                terminal.innerHTML = "";
                updateProgressBar(0);
                document.getElementById('progress-text-est').innerHTML = "Est: <span class='text-blue-600'>Calculating...</span>";
                document.getElementById('modal-success-btn-group').classList.add('hidden');
            
                const topSpinner = document.getElementById('modal-top-spinner');
                const topIcon = document.getElementById('modal-top-icon');
                topSpinner.className = "absolute inset-0 rounded-full border-2 border-blue-500 border-t-transparent animate-spin transition-colors";
                topSpinner.style.backgroundColor = "transparent";
                topIcon.className = "ph-fill ph-cpu text-3xl text-blue-600 z-10 relative";
            
                progressModal.classList.remove('hidden');
                progressModal.classList.add('flex');
                setTimeout(() => {
                    progressModal.classList.remove('opacity-0');
                    modalBox.classList.remove('scale-95');
                }, 10);
            
                // ==============================================================
                // 🚀 REAL-TIME TRACKING (Polling the Backend)
                // ==============================================================
                let isProcessing = true;
                let lastLog = "";
            
                // Ye interval har 1 second mein backend se status mangega
                const statusInterval = setInterval(async () => {
                    if (!isProcessing) return;
                    
                    try {
                        const statusRes = await fetch(`/api/progress?task_id=${taskId}`);
                        const statusData = await statusRes.json();
                        
                        if (statusData.percent !== undefined) {
                            updateProgressBar(statusData.percent);
                        }
                        
                        // Agar naya log aaya hai tabhi terminal mein dikhao
                        if (statusData.log && statusData.log !== lastLog) {
                            appendTerminalStep(statusData.log);
                            lastLog = statusData.log;
                        }
                    } catch (e) {
                        console.log("Waiting for server response...");
                    }
                }, 1000);
            
                // 🚀 SENDING THE ACTUAL POST REQUEST
                try {
                    const fetchPromise = fetch('/api/process-upscale', {
                        method: 'POST',
                        body: formData
                    });
                
                    const response = await fetchPromise;
                    const data = await response.json();
                    
                    // Processing done, stop asking for status
                    isProcessing = false;
                    clearInterval(statusInterval);
                
                    if (data.status === 'success') {
                        updateProgressBar(100);
                        appendTerminalStep("[✓] Final Masterpiece Generated and Exported.");
                    
                        const allSteps = terminal.querySelectorAll('div');
                        allSteps.forEach(s => {
                            const icon = s.querySelector('i');
                            if(icon && icon.classList.contains('ph-spinner')) {
                                icon.className = "ph-bold ph-check text-emerald-400";
                            }
                        });
                    
                        const finalStep = document.createElement('div');
                        finalStep.className = "flex items-center gap-2 mt-2";
                        finalStep.innerHTML = `<i class="ph-bold ph-check-circle text-emerald-500 text-lg"></i> <span class="text-emerald-400 font-bold typewriter-text">Process Complete. Masterpiece Ready.</span>`;
                        terminal.appendChild(finalStep);
                        terminal.scrollTop = terminal.scrollHeight;
                    
                        document.getElementById('progress-text-est').innerHTML = "<span class='text-emerald-600'>Done!</span>";
                        document.getElementById('modal-success-btn-group').classList.remove('hidden');
                    
                        topSpinner.classList.remove('animate-spin', 'border-t-transparent', 'border-blue-500');
                        topSpinner.classList.add('border-emerald-500');
                        topSpinner.style.backgroundColor = '#10b981';
                        topIcon.className = "ph-bold ph-check text-3xl text-white z-10 relative";
                    
                        currentProcessedImageUrl = data.output_path; 
                        currentMasterImageUrl = data.master_file;     
                        currentProcessedFilename = data.filename;     
                    
                        if (currentProcessedImageUrl.startsWith('data:image')) {
                            document.getElementById('upscaled-img').src = currentProcessedImageUrl;
                        } else {
                            document.getElementById('upscaled-img').src = currentProcessedImageUrl + "?t=" + new Date().getTime(); 
                        }
                    
                        if(data.resolution) {
                            document.getElementById('res-out').innerText = data.resolution + " (AI)";
                        }
                    } else {
                        appendTerminalError(data.message);
                        document.getElementById('progress-text-est').innerHTML = "<span class='text-red-500'>Failed</span>";
                        document.getElementById('progress-bar-fill').classList.replace('from-blue-500', 'from-red-500');
                        document.getElementById('progress-bar-fill').classList.replace('to-purple-500', 'to-red-600');
                    }
                
                } catch (error) {
                    isProcessing = false;
                    clearInterval(statusInterval);
                    appendTerminalError("Server connection failed. Is app.py running?");
                    document.getElementById('progress-text-est').innerHTML = "<span class='text-red-500'>Error</span>";
                }
            }
        
            function appendTerminalStep(text) {
                const step = document.createElement('div');
                step.className = "flex items-center gap-2 transform transition-all duration-300 translate-y-0 opacity-100";
                step.innerHTML = `<i class="ph-bold ph-spinner animate-spin text-blue-400"></i> <span class="typewriter-text">${text}</span>`;
                terminal.appendChild(step);
                terminal.scrollTop = terminal.scrollHeight;
            }
        
            function appendTerminalError(text) {
                const step = document.createElement('div');
                step.className = "flex items-center gap-2 text-red-400 mt-2";
                step.innerHTML = `<i class="ph-bold ph-warning-circle text-lg"></i> <span class="font-bold">SYSTEM ERROR: ${text}</span>`;
                terminal.appendChild(step);
                terminal.scrollTop = terminal.scrollHeight;
            }
        
            function updateProgressBar(percent) {
                document.getElementById('progress-bar-fill').style.width = percent + '%';
                document.getElementById('progress-text-percent').innerText = percent + '% Complete';
            }
        
            function closeProgressModal() {
                progressModal.classList.add('opacity-0');
                modalBox.classList.add('scale-95');
                setTimeout(() => {
                    progressModal.classList.add('hidden');
                    progressModal.classList.remove('flex');
                }, 300);
            }
        
            function resetCanvas() {
                previewState.classList.add('hidden');
                previewState.classList.remove('flex');
                bottomBar.classList.add('hidden');
                bottomBar.classList.remove('flex');
                uploadState.classList.remove('hidden');
                uploadState.classList.add('flex');
                document.getElementById('file-input').value = "";
                document.getElementById('file-name-display').innerText = "No Image Loaded";
                document.getElementById('res-in').innerText = "--";
                document.getElementById('res-out').innerText = "--";
                currentProcessedImageUrl = null;
                currentZoom = 100;
                updateZoomUI();
            }
        

        
            function goToExportTab() {
                closeProgressModal();
                const exportTabBtn = document.getElementById('export-tab-btn');
                if (exportTabBtn) {
                    switchTab('export', exportTabBtn);
                }
            }
            
            function saveColabUrl() {
                const url = document.getElementById('colab-url-input').value.trim();
                const btn = document.querySelector('button[onclick="saveColabUrl()"]');
                
                if(url) {
                    localStorage.setItem('nexus_colab_url', url);
                    
                    // Show small domain name in the dropdown button
                    try {
                        const domain = new URL(url).hostname;
                        document.getElementById('procmode-selected-text').innerHTML = `<i class="ph-fill ph-cloud text-blue-500 mr-1"></i> Cloud: ${domain}`;
                    } catch(e) {
                        document.getElementById('procmode-selected-text').innerText = "External Cloud GPU";
                    }
                    
                    // Show success on button temporarily
                    const oldHtml = btn.innerHTML;
                    btn.innerHTML = '<i class="ph-bold ph-check"></i> Connected Successfully!';
                    btn.classList.replace('from-blue-600', 'from-emerald-500');
                    btn.classList.replace('to-purple-600', 'to-emerald-600');
                    
                    setTimeout(() => {
                        document.getElementById('colab-modal').classList.add('hidden');
                        btn.innerHTML = oldHtml;
                        btn.classList.replace('from-emerald-500', 'from-blue-600');
                        btn.classList.replace('to-emerald-600', 'to-purple-600');
                    }, 1200);
                    
                } else {
                    localStorage.removeItem('nexus_colab_url');
                    document.getElementById('procmode-selected-text').innerText = document.getElementById('live-gpu-name').innerText.includes('Off') ? 'CPU Precision (Slow)' : 'GPU Accelerated (Fast)';
                    document.getElementById('colab-modal').classList.add('hidden');
                }
            }
            
            // On load, populate colab input if exists
            window.addEventListener('DOMContentLoaded', () => {
                const savedUrl = localStorage.getItem('nexus_colab_url');
                if(savedUrl && document.getElementById('colab-url-input')) {
                    document.getElementById('colab-url-input').value = savedUrl;
                    
                    // Auto-activate Cloud GPU mode if saved
                    const textSpan = document.getElementById('procmode-selected-text');
                    if (textSpan) {
                        try {
                            const domain = new URL(savedUrl).hostname;
                            textSpan.innerHTML = `<i class="ph-fill ph-cloud text-blue-500 mr-1"></i> Cloud: ${domain}`;
                        } catch(e) {
                            textSpan.innerHTML = 'External Cloud GPU';
                        }
                    }
                }
            });
        