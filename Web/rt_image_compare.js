import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ====================================================================================================
// [RT] Interactive Two Image Compare Extension
// Pixel-perfect Canvas engine with interactive slide wipe, wheel zoom, pan, difference, side-by-side,
// and synchronized fullscreen lightbox inspection.
// 1 Image Space for Slide, Click, Difference, Fade; Double Image Space ONLY for Side-by-Side.
// 100% English ASCII Only
// ====================================================================================================

app.registerExtension({
    name: "RareTutor.ImageCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RT_Image_Compare" || nodeData.name === "RTImageCompare") {

            // Style helper for node colors
            const applyNodeStyle = (node) => {
                node.color = "#004322"; // header color
                node.bgcolor = "#002211"; // body color
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) onConfigure.apply(this, arguments);
                applyNodeStyle(this);
                // Sanity clamp height to prevent runaway expansion from previous sessions
                if (this.size && (this.size[1] > 800 || this.size[1] < 300)) {
                    this.size[1] = 460;
                }
                if (this.size && (this.size[0] > 1400 || this.size[0] < 400)) {
                    this.size[0] = 560;
                }
                if (this.properties?.cached_images_a && this.properties?.cached_images_b && this.updateImages) {
                    this.updateImages(
                        this.properties.cached_images_a,
                        this.properties.cached_images_b,
                        this.properties.cached_mode,
                        this.properties.cached_dir
                    );
                }
                setTimeout(() => applyNodeStyle(this), 50);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                applyNodeStyle(this);
                setTimeout(() => applyNodeStyle(this), 50);

                // Set clean default size for comparison (matches standard 16:9 1-image space)
                if (!this.size || this.size[0] < 500 || this.size[1] < 380 || this.size[1] > 800) {
                    this.size = [560, 460];
                }

                // Default initial mode is Slide (Wipe)
                this.rtCompareState = {
                    mode: "slide", // "slide", "diff", "click", "side", "fade"
                    splitPos: 0.5,
                    direction: "horizontal", // "horizontal" or "vertical"
                    diffGain: 1, // 1x, 2x, 5x, 10x
                    activeLayer: "a",
                    batchIndex: 0,
                    swapped: false,
                    zoom: 1.0,
                    panX: 0,
                    panY: 0,
                    imagesA: [],
                    imagesB: [],
                    imgAObj: null,
                    imgBObj: null,
                    isLoaded: false,
                };

                // Master Container Widget
                const container = document.createElement("div");
                container.className = "rt-image-compare-container";
                Object.assign(container.style, {
                    boxSizing: "border-box",
                    width: "calc(100% - 16px)",
                    height: "335px",
                    margin: "0 auto 10px auto",
                    display: "flex",
                    flexDirection: "column",
                    position: "relative",
                    userSelect: "none",
                    overflow: "hidden",
                    borderRadius: "8px",
                    backgroundColor: "#001a0d",
                    boxShadow: "inset 0 0 0 1px #004322, 0 4px 14px rgba(0, 34, 17, 0.4)",
                });

                // Shield events from LiteGraph canvas navigation
                const stopEvent = (e) => e.stopPropagation();
                container.addEventListener("pointerdown", stopEvent);
                container.addEventListener("mousedown", stopEvent);
                container.addEventListener("contextmenu", (e) => e.preventDefault());

                // 1. Stage with HTML5 Canvas
                const stage = document.createElement("div");
                Object.assign(stage.style, {
                    position: "relative",
                    flex: "1 1 auto",
                    width: "100%",
                    height: "100%",
                    minHeight: "220px",
                    overflow: "hidden",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundColor: "#000000",
                    cursor: "ew-resize",
                });

                const canvas = document.createElement("canvas");
                Object.assign(canvas.style, {
                    display: "block",
                    width: "100%",
                    height: "100%",
                });
                stage.appendChild(canvas);

                // Badges
                const badgeA = document.createElement("div");
                badgeA.innerText = "A";
                Object.assign(badgeA.style, {
                    position: "absolute",
                    top: "10px",
                    left: "10px",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    backgroundColor: "rgba(0, 34, 17, 0.85)",
                    border: "1px solid rgba(0, 229, 255, 0.4)",
                    color: "#00e5ff",
                    fontSize: "11px",
                    fontWeight: "700",
                    zIndex: "15",
                    pointerEvents: "none",
                    display: "none",
                });
                stage.appendChild(badgeA);

                const badgeB = document.createElement("div");
                badgeB.innerText = "B";
                Object.assign(badgeB.style, {
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    backgroundColor: "rgba(0, 34, 17, 0.85)",
                    border: "1px solid rgba(255, 0, 127, 0.4)",
                    color: "#ff007f",
                    fontSize: "11px",
                    fontWeight: "700",
                    zIndex: "15",
                    pointerEvents: "none",
                    display: "none",
                });
                stage.appendChild(badgeB);

                const modeBadge = document.createElement("div");
                modeBadge.innerText = "Mode: Slide";
                Object.assign(modeBadge.style, {
                    position: "absolute",
                    top: "10px",
                    left: "50%",
                    transform: "translateX(-50%)",
                    padding: "4px 10px",
                    borderRadius: "4px",
                    backgroundColor: "rgba(0, 34, 17, 0.9)",
                    border: "1px solid rgba(255, 255, 255, 0.25)",
                    color: "#ffffff",
                    fontSize: "11px",
                    fontWeight: "600",
                    zIndex: "15",
                    pointerEvents: "none",
                    display: "none",
                });
                stage.appendChild(modeBadge);

                // Zoom Reset Pill Badge
                const zoomBadge = document.createElement("div");
                zoomBadge.innerText = "1.0x";
                Object.assign(zoomBadge.style, {
                    position: "absolute",
                    bottom: "10px",
                    right: "10px",
                    padding: "3px 8px",
                    borderRadius: "4px",
                    backgroundColor: "rgba(43, 108, 176, 0.9)",
                    border: "1px solid rgba(255, 255, 255, 0.3)",
                    color: "#ffffff",
                    fontSize: "10px",
                    fontWeight: "700",
                    zIndex: "15",
                    cursor: "pointer",
                    display: "none",
                });
                zoomBadge.title = "Click to reset zoom to 100%";
                zoomBadge.addEventListener("click", (e) => {
                    e.stopPropagation();
                    this.rtCompareState.zoom = 1.0;
                    this.rtCompareState.panX = 0;
                    this.rtCompareState.panY = 0;
                    renderCanvas();
                });
                stage.appendChild(zoomBadge);

                // 2. Toolbar
                const toolbar = document.createElement("div");
                Object.assign(toolbar.style, {
                    height: "38px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "0 10px",
                    backgroundColor: "#002b16",
                    borderTop: "1px solid rgba(0, 67, 34, 0.6)",
                    zIndex: "20",
                });

                // Mode Button Group
                const modeGroup = document.createElement("div");
                Object.assign(modeGroup.style, {
                    display: "flex",
                    gap: "4px",
                });

                const modes = [
                    { id: "slide", label: "Slide" },
                    { id: "diff", label: "Difference" },
                    { id: "click", label: "Click" },
                    { id: "side", label: "Side-by-Side" },
                    { id: "fade", label: "Fade" },
                ];

                const modeButtons = {};
                modes.forEach(m => {
                    const btn = document.createElement("button");
                    btn.innerText = m.label;
                    const isActive = m.id === "slide";
                    Object.assign(btn.style, {
                        backgroundColor: isActive ? "#2b6cb0" : "transparent",
                        color: isActive ? "#ffffff" : "#a0aec0",
                        border: "none",
                        borderRadius: "4px",
                        padding: "4px 9px",
                        fontSize: "11px",
                        fontWeight: "600",
                        cursor: "pointer",
                        outline: "none",
                    });

                    btn.addEventListener("click", () => {
                        this.rtCompareState.mode = m.id;
                        this.rtCompareState.zoom = 1.0;
                        this.rtCompareState.panX = 0;
                        this.rtCompareState.panY = 0;
                        updateToolbarButtons();
                        renderCanvas();
                    });
                    modeGroup.appendChild(btn);
                    modeButtons[m.id] = btn;
                });
                toolbar.appendChild(modeGroup);

                // Actions Group
                const actionGroup = document.createElement("div");
                Object.assign(actionGroup.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                });

                // Gain Boost Button for Difference Mode
                const boostBtn = document.createElement("button");
                boostBtn.innerText = "Gain: 1x";
                Object.assign(boostBtn.style, {
                    backgroundColor: "rgba(255, 255, 255, 0.08)",
                    color: "#ecc94b",
                    border: "none",
                    borderRadius: "4px",
                    padding: "3px 7px",
                    fontSize: "11px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "none",
                });
                boostBtn.addEventListener("click", () => {
                    const gains = [1, 2, 5, 10];
                    const nextIdx = (gains.indexOf(this.rtCompareState.diffGain) + 1) % gains.length;
                    this.rtCompareState.diffGain = gains[nextIdx];
                    boostBtn.innerText = `Gain: ${this.rtCompareState.diffGain}x`;
                    renderCanvas();
                });
                actionGroup.appendChild(boostBtn);

                // Batch Stepper Counter
                const batchLabel = document.createElement("span");
                batchLabel.innerText = "1/1";
                Object.assign(batchLabel.style, {
                    fontSize: "11px",
                    color: "#81e6d9",
                    fontWeight: "600",
                });

                const prevBtn = document.createElement("button");
                prevBtn.innerText = "<";
                const nextBtn = document.createElement("button");
                nextBtn.innerText = ">";
                [prevBtn, nextBtn].forEach(b => {
                    Object.assign(b.style, {
                        backgroundColor: "rgba(255, 255, 255, 0.08)",
                        color: "#cbd5e0",
                        border: "none",
                        borderRadius: "3px",
                        padding: "2px 6px",
                        fontSize: "10px",
                        cursor: "pointer",
                    });
                });

                prevBtn.addEventListener("click", () => {
                    if (this.rtCompareState.batchIndex > 0) {
                        this.rtCompareState.batchIndex--;
                        loadCurrentImages();
                    }
                });
                nextBtn.addEventListener("click", () => {
                    const max = Math.min(this.rtCompareState.imagesA.length, this.rtCompareState.imagesB.length) - 1;
                    if (this.rtCompareState.batchIndex < max) {
                        this.rtCompareState.batchIndex++;
                        loadCurrentImages();
                    }
                });

                actionGroup.appendChild(prevBtn);
                actionGroup.appendChild(batchLabel);
                actionGroup.appendChild(nextBtn);

                // Swap A <-> B Button
                const swapBtn = document.createElement("button");
                swapBtn.innerText = "Swap";
                Object.assign(swapBtn.style, {
                    backgroundColor: "rgba(255, 255, 255, 0.08)",
                    color: "#cbd5e0",
                    border: "none",
                    borderRadius: "4px",
                    padding: "3px 7px",
                    fontSize: "11px",
                    cursor: "pointer",
                });
                swapBtn.addEventListener("click", () => {
                    this.rtCompareState.swapped = !this.rtCompareState.swapped;
                    loadCurrentImages();
                });
                actionGroup.appendChild(swapBtn);

                // Fullscreen Lightbox Button
                const zoomBtn = document.createElement("button");
                zoomBtn.innerText = "Zoom Window";
                Object.assign(zoomBtn.style, {
                    backgroundColor: "#2b6cb0",
                    color: "#ffffff",
                    border: "none",
                    borderRadius: "4px",
                    padding: "3px 8px",
                    fontSize: "11px",
                    fontWeight: "600",
                    cursor: "pointer",
                });
                zoomBtn.addEventListener("click", () => openFullscreenLightbox());
                actionGroup.appendChild(zoomBtn);

                toolbar.appendChild(actionGroup);

                container.appendChild(stage);
                container.appendChild(toolbar);

                const domWidget = this.addDOMWidget("rt_image_compare_widget", "div", container, {
                    serialize: false,
                    hideOnZoom: false,
                });
                // Fixed base computeSize - prevents infinite recursive vertical expansion loop
                domWidget.computeSize = (width) => [width, 345];

                // Responsive node resizing only when manually dragged by user
                const onResize = this.onResize;
                this.onResize = function (size) {
                    if (onResize) onResize.apply(this, arguments);
                    if (size && size[1]) {
                        container.style.height = Math.max(220, Math.round(size[1] - 135)) + "px";
                    }
                    renderCanvas();
                };

                // --- 3. Pixel-Perfect HTML5 Canvas Rendering Engine ---
                const renderCanvasToTarget = (targetCanvas, state) => {
                    const ctx = targetCanvas.getContext("2d", { willReadFrequently: true });
                    const dpr = window.devicePixelRatio || 1;
                    const w = targetCanvas.width;
                    const h = targetCanvas.height;

                    ctx.globalAlpha = 1.0;
                    ctx.globalCompositeOperation = "source-over";
                    ctx.clearRect(0, 0, w, h);

                    if (!state.isLoaded || !state.imgAObj || !state.imgBObj) {
                        ctx.fillStyle = "#001a0d";
                        ctx.fillRect(0, 0, w, h);
                        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
                        ctx.font = `${14 * dpr}px sans-serif`;
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText("Connect Image A & Image B and Run Workflow", w / 2, h / 2);
                        return;
                    }

                    const imgA = state.swapped ? state.imgBObj : state.imgAObj;
                    const imgB = state.swapped ? state.imgAObj : state.imgBObj;

                    const m = state.mode;
                    const isSide = (m === "side");

                    // Base aspect ratio:
                    // ONLY in Side-by-Side mode: DOUBLE image space (width = widthA + widthB)
                    // In all other modes (Slide, Difference, Click, Fade): 1 SINGLE IMAGE SPACE!
                    const singleAspect = imgA.naturalWidth / imgA.naturalHeight;
                    const targetAspect = isSide ? (singleAspect * 2) : singleAspect;

                    const canvasAspect = w / h;
                    let dw = w;
                    let dh = h;
                    let dx = 0;
                    let dy = 0;

                    if (canvasAspect > targetAspect) {
                        dh = h;
                        dw = Math.round(h * targetAspect);
                        dx = Math.round((w - dw) / 2);
                    } else {
                        dw = w;
                        dh = Math.round(w / targetAspect);
                        dy = Math.round((h - dh) / 2);
                    }

                    // Store active base rect on canvas for coordinate conversions
                    targetCanvas.imageRect = { dx, dy, dw, dh, isSide };

                    // Effective zoomed & panned coordinates
                    const zoom = state.zoom || 1.0;
                    const panX = state.panX || 0;
                    const panY = state.panY || 0;

                    const curDw = Math.round(dw * zoom);
                    const curDh = Math.round(dh * zoom);
                    const curDx = Math.round(dx + panX);
                    const curDy = Math.round(dy + panY);

                    // Viewport clip: contain within canvas bounds
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(0, 0, w, h);
                    ctx.clip();

                    ctx.fillStyle = "#000000";
                    ctx.fillRect(0, 0, w, h);

                    // Solid neutral opaque base behind image bounds
                    ctx.fillStyle = "#141414";
                    ctx.fillRect(curDx, curDy, curDw, curDh);

                    if (isSide) {
                        // =========================================================
                        // DOUBLE IMAGE SPACE: Show two full images side-by-side
                        // =========================================================
                        const halfW = Math.round(curDw / 2);
                        const leftW = halfW;
                        const rightW = curDw - halfW;

                        // Image A (Left Space)
                        ctx.drawImage(imgA, curDx, curDy, leftW, curDh);

                        // Image B (Right Space)
                        ctx.drawImage(imgB, curDx + leftW, curDy, rightW, curDh);

                        // Divider line separating the two image spaces
                        const divX = curDx + leftW;
                        const lineY1 = Math.max(0, curDy);
                        const lineY2 = Math.min(h, curDy + curDh);
                        if (divX >= 0 && divX <= w && lineY2 > lineY1) {
                            ctx.beginPath();
                            ctx.moveTo(divX, lineY1);
                            ctx.lineTo(divX, lineY2);
                            ctx.lineWidth = 2 * dpr;
                            ctx.strokeStyle = "#ffffff";
                            ctx.shadowColor = "rgba(0,0,0,0.85)";
                            ctx.shadowBlur = 4;
                            ctx.stroke();
                            ctx.shadowBlur = 0;
                        }

                    } else if (m === "slide") {
                        // =========================================================
                        // 1 IMAGE SPACE: Slide Wipe between overlaid images
                        // Neither image is drawn underneath the other to prevent transparency bleed
                        // =========================================================
                        const isVert = state.direction === "vertical";
                        if (isVert) {
                            const splitY = curDy + Math.round(curDh * state.splitPos);

                            // Top portion: Image A
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(curDx, curDy, curDw, Math.max(0, splitY - curDy));
                            ctx.clip();
                            ctx.drawImage(imgA, curDx, curDy, curDw, curDh);
                            ctx.restore();

                            // Bottom portion: Image B
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(curDx, splitY, curDw, Math.max(0, (curDy + curDh) - splitY));
                            ctx.clip();
                            ctx.drawImage(imgB, curDx, curDy, curDw, curDh);
                            ctx.restore();

                            const lineX1 = Math.max(0, curDx);
                            const lineX2 = Math.min(w, curDx + curDw);
                            if (splitY >= 0 && splitY <= h && lineX2 > lineX1) {
                                ctx.beginPath();
                                ctx.moveTo(lineX1, splitY);
                                ctx.lineTo(lineX2, splitY);
                                ctx.lineWidth = 2 * dpr;
                                ctx.strokeStyle = "#ffffff";
                                ctx.shadowColor = "rgba(0,0,0,0.85)";
                                ctx.shadowBlur = 5;
                                ctx.stroke();
                                ctx.shadowBlur = 0;

                                const knobX = Math.max(lineX1 + 18 * dpr, Math.min(lineX2 - 18 * dpr, curDx + curDw / 2));
                                ctx.beginPath();
                                ctx.arc(knobX, splitY, 13 * dpr, 0, Math.PI * 2);
                                ctx.fillStyle = "#002b16";
                                ctx.fill();
                                ctx.lineWidth = 2 * dpr;
                                ctx.strokeStyle = "#ffffff";
                                ctx.stroke();
                                ctx.fillStyle = "#ffffff";
                                ctx.font = `bold ${9 * dpr}px sans-serif`;
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText("^v", knobX, splitY);
                            }
                        } else {
                            const splitX = curDx + Math.round(curDw * state.splitPos);

                            // Left portion: Image A
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(curDx, curDy, Math.max(0, splitX - curDx), curDh);
                            ctx.clip();
                            ctx.drawImage(imgA, curDx, curDy, curDw, curDh);
                            ctx.restore();

                            // Right portion: Image B
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(splitX, curDy, Math.max(0, (curDx + curDw) - splitX), curDh);
                            ctx.clip();
                            ctx.drawImage(imgB, curDx, curDy, curDw, curDh);
                            ctx.restore();

                            const lineY1 = Math.max(0, curDy);
                            const lineY2 = Math.min(h, curDy + curDh);
                            if (splitX >= 0 && splitX <= w && lineY2 > lineY1) {
                                ctx.beginPath();
                                ctx.moveTo(splitX, lineY1);
                                ctx.lineTo(splitX, lineY2);
                                ctx.lineWidth = 2 * dpr;
                                ctx.strokeStyle = "#ffffff";
                                ctx.shadowColor = "rgba(0,0,0,0.85)";
                                ctx.shadowBlur = 5;
                                ctx.stroke();
                                ctx.shadowBlur = 0;

                                const knobY = Math.max(lineY1 + 18 * dpr, Math.min(lineY2 - 18 * dpr, curDy + curDh / 2));
                                ctx.beginPath();
                                ctx.arc(splitX, knobY, 13 * dpr, 0, Math.PI * 2);
                                ctx.fillStyle = "#002b16";
                                ctx.fill();
                                ctx.lineWidth = 2 * dpr;
                                ctx.strokeStyle = "#ffffff";
                                ctx.stroke();
                                ctx.fillStyle = "#ffffff";
                                ctx.font = `bold ${9 * dpr}px sans-serif`;
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText("<>", splitX, knobY);
                            }
                        }

                    } else if (m === "click") {
                        // =========================================================
                        // 1 IMAGE SPACE: Click to toggle A or B in same space
                        // =========================================================
                        const target = state.activeLayer === "a" ? imgA : imgB;
                        ctx.drawImage(target, curDx, curDy, curDw, curDh);

                    } else if (m === "diff") {
                        // =========================================================
                        // 1 IMAGE SPACE: Pure Mathematical Difference (|A - B|)
                        // =========================================================
                        ctx.drawImage(imgB, curDx, curDy, curDw, curDh);
                        ctx.globalCompositeOperation = "difference";
                        ctx.drawImage(imgA, curDx, curDy, curDw, curDh);
                        ctx.globalCompositeOperation = "source-over";

                        if (state.diffGain > 1) {
                            try {
                                const vx = Math.max(0, curDx);
                                const vy = Math.max(0, curDy);
                                const vw = Math.min(w, curDx + curDw) - vx;
                                const vh = Math.min(h, curDy + curDh) - vy;
                                if (vw > 0 && vh > 0) {
                                    const imgData = ctx.getImageData(vx, vy, vw, vh);
                                    const d = imgData.data;
                                    const g = state.diffGain;
                                    for (let i = 0; i < d.length; i += 4) {
                                        d[i] = Math.min(255, d[i] * g);
                                        d[i + 1] = Math.min(255, d[i + 1] * g);
                                        d[i + 2] = Math.min(255, d[i + 2] * g);
                                    }
                                    ctx.putImageData(imgData, vx, vy);
                                }
                            } catch (e) {}
                        }

                    } else if (m === "fade") {
                        // =========================================================
                        // 1 IMAGE SPACE: Alpha Dissolve Blend in same space
                        // =========================================================
                        ctx.drawImage(imgB, curDx, curDy, curDw, curDh);
                        ctx.globalAlpha = Math.max(0, Math.min(1, 1.0 - state.splitPos));
                        ctx.drawImage(imgA, curDx, curDy, curDw, curDh);
                        ctx.globalAlpha = 1.0;
                    }

                    ctx.restore();
                };

                const renderCanvas = () => {
                    const rect = stage.getBoundingClientRect();
                    const dpr = window.devicePixelRatio || 1;
                    const w = Math.max(50, Math.round(rect.width * dpr));
                    const h = Math.max(50, Math.round(rect.height * dpr));

                    if (canvas.width !== w || canvas.height !== h) {
                        canvas.width = w;
                        canvas.height = h;
                    }

                    renderCanvasToTarget(canvas, this.rtCompareState);
                    updateBadges();
                };

                const updateBadges = () => {
                    const state = this.rtCompareState;
                    if (!state.isLoaded) {
                        badgeA.style.display = "none";
                        badgeB.style.display = "none";
                        modeBadge.style.display = "none";
                        zoomBadge.style.display = "none";
                        return;
                    }

                    const dataA = state.swapped ? state.imagesB[state.batchIndex] : state.imagesA[state.batchIndex];
                    const dataB = state.swapped ? state.imagesA[state.batchIndex] : state.imagesB[state.batchIndex];

                    const labelA = state.swapped ? "B" : "A";
                    const labelB = state.swapped ? "A" : "B";

                    badgeA.innerText = `${labelA} (${dataA?.width || ""}x${dataA?.height || ""})`;
                    badgeB.innerText = `${labelB} (${dataB?.width || ""}x${dataB?.height || ""})`;

                    if (state.mode === "click") {
                        badgeA.style.display = state.activeLayer === "a" ? "block" : "none";
                        badgeB.style.display = state.activeLayer === "b" ? "block" : "none";
                    } else {
                        badgeA.style.display = "block";
                        badgeB.style.display = "block";
                    }

                    modeBadge.style.display = "block";
                    if (state.mode === "diff") {
                        modeBadge.innerText = `Difference (${state.diffGain}x)`;
                    } else if (state.mode === "click") {
                        modeBadge.innerText = `Showing: ${state.activeLayer === "a" ? labelA : labelB}`;
                    } else if (state.mode === "slide") {
                        modeBadge.innerText = `Slide Wipe (${Math.round(state.splitPos * 100)}%)`;
                    } else if (state.mode === "side") {
                        modeBadge.innerText = "Side-by-Side (A | B)";
                    } else if (state.mode === "fade") {
                        modeBadge.innerText = `Fade (${Math.round((1.0 - state.splitPos) * 100)}% A)`;
                    }

                    // Zoom indicator
                    if (state.zoom > 1.01) {
                        zoomBadge.innerText = `${state.zoom.toFixed(1)}x (Reset)`;
                        zoomBadge.style.display = "block";
                    } else {
                        zoomBadge.style.display = "none";
                    }
                };

                const updateToolbarButtons = () => {
                    const m = this.rtCompareState.mode;
                    Object.keys(modeButtons).forEach(k => {
                        const active = k === m;
                        modeButtons[k].style.backgroundColor = active ? "#2b6cb0" : "transparent";
                        modeButtons[k].style.color = active ? "#ffffff" : "#a0aec0";
                    });

                    boostBtn.style.display = m === "diff" ? "inline-block" : "none";

                    // Cursor adjustment
                    if (this.rtCompareState.zoom > 1.01) {
                        stage.style.cursor = "grab";
                    } else if (m === "slide" || m === "fade") {
                        stage.style.cursor = this.rtCompareState.direction === "vertical" ? "ns-resize" : "ew-resize";
                    } else if (m === "click") {
                        stage.style.cursor = "pointer";
                    } else {
                        stage.style.cursor = "default";
                    }
                };

                // --- 4. Interactive Pointer & Mouse Wheel Zoom Engine ---
                const handlePointerPos = (clientX, clientY, targetCanvas, state, callback) => {
                    const rect = targetCanvas.getBoundingClientRect();
                    if (!rect.width || !rect.height) return;

                    const scaleX = targetCanvas.width / rect.width;
                    const scaleY = targetCanvas.height / rect.height;
                    const canvasX = (clientX - rect.left) * scaleX;
                    const canvasY = (clientY - rect.top) * scaleY;

                    const r = targetCanvas.imageRect;
                    if (!r || !r.dw || !r.dh) return;

                    const zoom = state.zoom || 1.0;
                    const curDw = r.dw * zoom;
                    const curDh = r.dh * zoom;
                    const curDx = r.dx + (state.panX || 0);
                    const curDy = r.dy + (state.panY || 0);

                    let pos;
                    if (state.direction === "vertical") {
                        pos = (canvasY - curDy) / curDh;
                    } else {
                        pos = (canvasX - curDx) / curDw;
                    }

                    pos = Math.max(0, Math.min(1, pos));
                    state.splitPos = pos;
                    if (callback) callback();
                };

                // Mouse wheel zoom centered on cursor
                const handleWheelZoom = (e, targetCanvas, state, callback) => {
                    e.preventDefault();
                    e.stopPropagation();

                    if (!state.isLoaded || !targetCanvas.imageRect) return;

                    const rect = targetCanvas.getBoundingClientRect();
                    if (!rect.width || !rect.height) return;

                    const scaleX = targetCanvas.width / rect.width;
                    const scaleY = targetCanvas.height / rect.height;
                    const mouseX = (e.clientX - rect.left) * scaleX;
                    const mouseY = (e.clientY - rect.top) * scaleY;

                    const r = targetCanvas.imageRect;
                    const curZoom = state.zoom || 1.0;
                    const curPanX = state.panX || 0;
                    const curPanY = state.panY || 0;

                    const curDw = r.dw * curZoom;
                    const curDh = r.dh * curZoom;
                    const curDx = r.dx + curPanX;
                    const curDy = r.dy + curPanY;

                    const u = (mouseX - curDx) / curDw;
                    const v = (mouseY - curDy) / curDh;

                    const zoomStep = e.deltaY < 0 ? 1.2 : (1 / 1.2);
                    let newZoom = curZoom * zoomStep;

                    if (newZoom <= 1.01) {
                        state.zoom = 1.0;
                        state.panX = 0;
                        state.panY = 0;
                    } else {
                        newZoom = Math.min(30.0, newZoom);
                        const newDw = r.dw * newZoom;
                        const newDh = r.dh * newZoom;
                        const newDx = mouseX - u * newDw;
                        const newDy = mouseY - v * newDh;

                        state.zoom = newZoom;
                        state.panX = newDx - r.dx;
                        state.panY = newDy - r.dy;
                    }

                    if (callback) callback();
                };

                // Pointer and Pan state
                let isDragging = false;
                let isPanning = false;
                let startX = 0;
                let startY = 0;
                let startPanX = 0;
                let startPanY = 0;

                stage.addEventListener("wheel", (e) => {
                    handleWheelZoom(e, canvas, this.rtCompareState, () => {
                        renderCanvas();
                        updateToolbarButtons();
                    });
                }, { passive: false });

                stage.addEventListener("dblclick", (e) => {
                    e.stopPropagation();
                    this.rtCompareState.zoom = 1.0;
                    this.rtCompareState.panX = 0;
                    this.rtCompareState.panY = 0;
                    renderCanvas();
                    updateToolbarButtons();
                });

                stage.addEventListener("pointerdown", (e) => {
                    const state = this.rtCompareState;
                    if (!state.isLoaded) return;

                    // Middle mouse, Right mouse, Shift+Left, or Left drag when zoomed in non-slide mode pans
                    if (e.button === 1 || e.button === 2 || (e.shiftKey && e.button === 0) || (state.zoom > 1.01 && state.mode !== "slide" && state.mode !== "fade" && e.button === 0)) {
                        isPanning = true;
                        const rect = canvas.getBoundingClientRect();
                        const scaleX = canvas.width / rect.width;
                        const scaleY = canvas.height / rect.height;
                        startX = e.clientX * scaleX;
                        startY = e.clientY * scaleY;
                        startPanX = state.panX || 0;
                        startPanY = state.panY || 0;
                        stage.style.cursor = "grabbing";
                        try { e.target.setPointerCapture(e.pointerId); } catch (_) {}
                        return;
                    }

                    if (e.button === 0) {
                        if (state.mode === "click") {
                            state.activeLayer = state.activeLayer === "a" ? "b" : "a";
                            renderCanvas();
                            return;
                        }

                        isDragging = true;
                        try { e.target.setPointerCapture(e.pointerId); } catch (_) {}
                        handlePointerPos(e.clientX, e.clientY, canvas, state, () => renderCanvas());
                    }
                });

                window.addEventListener("pointermove", (e) => {
                    const state = this.rtCompareState;
                    if (isPanning) {
                        const rect = canvas.getBoundingClientRect();
                        const scaleX = canvas.width / rect.width;
                        const scaleY = canvas.height / rect.height;
                        const curX = e.clientX * scaleX;
                        const curY = e.clientY * scaleY;
                        state.panX = startPanX + (curX - startX);
                        state.panY = startPanY + (curY - startY);
                        renderCanvas();
                        return;
                    }

                    if (isDragging) {
                        handlePointerPos(e.clientX, e.clientY, canvas, state, () => renderCanvas());
                    }
                });

                window.addEventListener("pointerup", (e) => {
                    if (isPanning) {
                        isPanning = false;
                        updateToolbarButtons();
                    }
                    if (isDragging) {
                        isDragging = false;
                    }
                    try { e.target.releasePointerCapture(e.pointerId); } catch (_) {}
                });

                stage.addEventListener("pointermove", (e) => {
                    const state = this.rtCompareState;
                    if (!isPanning && !isDragging && (state.mode === "slide" || state.mode === "fade")) {
                        handlePointerPos(e.clientX, e.clientY, canvas, state, () => renderCanvas());
                    }
                });

                // Auto-redraw on resize
                const resizeObserver = new ResizeObserver(() => renderCanvas());
                resizeObserver.observe(stage);

                // --- 5. Image Loader Engine ---
                const loadCurrentImages = () => {
                    const state = this.rtCompareState;
                    if (!state.imagesA.length || !state.imagesB.length) return;

                    const idx = Math.max(0, Math.min(state.batchIndex, state.imagesA.length - 1));
                    const dataA = state.imagesA[idx];
                    const dataB = state.imagesB[idx];

                    batchLabel.innerText = `${idx + 1}/${Math.min(state.imagesA.length, state.imagesB.length)}`;

                    const urlA = api.apiURL(`/view?filename=${encodeURIComponent(dataA.filename)}&type=${dataA.type}&subfolder=${dataA.subfolder || ""}`);
                    const urlB = api.apiURL(`/view?filename=${encodeURIComponent(dataB.filename)}&type=${dataB.type}&subfolder=${dataB.subfolder || ""}`);

                    let loadedCount = 0;
                    const checkDone = () => {
                        loadedCount++;
                        if (loadedCount >= 2) {
                            state.isLoaded = true;
                            updateToolbarButtons();
                            renderCanvas();
                        }
                    };

                    const imgA = new Image();
                    imgA.onload = checkDone;
                    imgA.onerror = (err) => {
                        console.error("[RT Image Compare] Failed to load Image A:", urlA, err);
                        checkDone();
                    };
                    imgA.src = urlA;
                    if (imgA.complete && imgA.naturalWidth) checkDone();
                    state.imgAObj = imgA;

                    const imgB = new Image();
                    imgB.onload = checkDone;
                    imgB.onerror = (err) => {
                        console.error("[RT Image Compare] Failed to load Image B:", urlB, err);
                        checkDone();
                    };
                    imgB.src = urlB;
                    if (imgB.complete && imgB.naturalWidth) checkDone();
                    state.imgBObj = imgB;
                };

                // --- 6. Fullscreen Lightbox Modal Window with Deep Zoom & Pan ---
                const openFullscreenLightbox = () => {
                    const modal = document.createElement("div");
                    Object.assign(modal.style, {
                        position: "fixed",
                        top: "0",
                        left: "0",
                        width: "100vw",
                        height: "100vh",
                        backgroundColor: "rgba(0, 15, 8, 0.97)",
                        backdropFilter: "blur(12px)",
                        zIndex: "99999",
                        display: "flex",
                        flexDirection: "column",
                        userSelect: "none",
                    });

                    // Modal Header with Controls
                    const modalHeader = document.createElement("div");
                    Object.assign(modalHeader.style, {
                        height: "48px",
                        padding: "0 16px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        backgroundColor: "#002b16",
                        borderBottom: "1px solid rgba(0, 67, 34, 0.8)",
                    });

                    const modalLeft = document.createElement("div");
                    Object.assign(modalLeft.style, {
                        display: "flex",
                        alignItems: "center",
                        gap: "12px",
                    });

                    const modalTitle = document.createElement("div");
                    modalTitle.innerText = "RT Image Compare (Interactive Inspection)";
                    Object.assign(modalTitle.style, {
                        color: "#ffffff",
                        fontSize: "13px",
                        fontWeight: "700",
                    });
                    modalLeft.appendChild(modalTitle);

                    // Modal Mode Switcher
                    const modalModeGroup = document.createElement("div");
                    Object.assign(modalModeGroup.style, {
                        display: "flex",
                        gap: "4px",
                    });

                    const mButtons = {};
                    modes.forEach(m => {
                        const btn = document.createElement("button");
                        btn.innerText = m.label;
                        const active = m.id === this.rtCompareState.mode;
                        Object.assign(btn.style, {
                            backgroundColor: active ? "#2b6cb0" : "rgba(255,255,255,0.06)",
                            color: active ? "#ffffff" : "#cbd5e0",
                            border: "none",
                            borderRadius: "4px",
                            padding: "4px 8px",
                            fontSize: "11px",
                            fontWeight: "600",
                            cursor: "pointer",
                        });
                        btn.addEventListener("click", () => {
                            this.rtCompareState.mode = m.id;
                            this.rtCompareState.zoom = 1.0;
                            this.rtCompareState.panX = 0;
                            this.rtCompareState.panY = 0;
                            updateModalButtons();
                            updateToolbarButtons();
                            redrawModal();
                            renderCanvas();
                        });
                        modalModeGroup.appendChild(btn);
                        mButtons[m.id] = btn;
                    });
                    modalLeft.appendChild(modalModeGroup);
                    modalHeader.appendChild(modalLeft);

                    const updateModalButtons = () => {
                        const curM = this.rtCompareState.mode;
                        Object.keys(mButtons).forEach(k => {
                            const act = k === curM;
                            mButtons[k].style.backgroundColor = act ? "#2b6cb0" : "rgba(255,255,255,0.06)";
                            mButtons[k].style.color = act ? "#ffffff" : "#cbd5e0";
                        });
                    };

                    const modalRight = document.createElement("div");
                    Object.assign(modalRight.style, {
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                    });

                    const resetZoomBtn = document.createElement("button");
                    resetZoomBtn.innerText = "1:1 Reset Zoom";
                    Object.assign(resetZoomBtn.style, {
                        backgroundColor: "rgba(255, 255, 255, 0.08)",
                        color: "#ecc94b",
                        border: "1px solid rgba(236, 201, 75, 0.4)",
                        borderRadius: "4px",
                        padding: "5px 10px",
                        fontSize: "11px",
                        fontWeight: "600",
                        cursor: "pointer",
                    });
                    resetZoomBtn.addEventListener("click", () => {
                        this.rtCompareState.zoom = 1.0;
                        this.rtCompareState.panX = 0;
                        this.rtCompareState.panY = 0;
                        redrawModal();
                        renderCanvas();
                    });
                    modalRight.appendChild(resetZoomBtn);

                    const closeBtn = document.createElement("button");
                    closeBtn.innerText = "Close (Esc)";
                    Object.assign(closeBtn.style, {
                        backgroundColor: "#e53e3e",
                        color: "#ffffff",
                        border: "none",
                        borderRadius: "4px",
                        padding: "5px 12px",
                        fontSize: "11px",
                        fontWeight: "600",
                        cursor: "pointer",
                    });

                    const closeModal = () => {
                        window.removeEventListener("keydown", onKeyDown);
                        if (modal.parentNode) document.body.removeChild(modal);
                        renderCanvas();
                    };

                    const onKeyDown = (e) => {
                        if (e.key === "Escape") closeModal();
                    };
                    window.addEventListener("keydown", onKeyDown);

                    closeBtn.addEventListener("click", closeModal);
                    modalRight.appendChild(closeBtn);
                    modalHeader.appendChild(modalRight);

                    // Modal Stage
                    const modalStage = document.createElement("div");
                    Object.assign(modalStage.style, {
                        flex: "1",
                        position: "relative",
                        overflow: "hidden",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        backgroundColor: "#000000",
                        cursor: "ew-resize",
                    });

                    const modalCanvas = document.createElement("canvas");
                    Object.assign(modalCanvas.style, {
                        display: "block",
                        width: "100%",
                        height: "100%",
                    });
                    modalStage.appendChild(modalCanvas);

                    modal.appendChild(modalHeader);
                    modal.appendChild(modalStage);
                    document.body.appendChild(modal);

                    const redrawModal = () => {
                        const rect = modalStage.getBoundingClientRect();
                        const dpr = window.devicePixelRatio || 1;
                        const w = Math.max(100, Math.round(rect.width * dpr));
                        const h = Math.max(100, Math.round(rect.height * dpr));
                        modalCanvas.width = w;
                        modalCanvas.height = h;
                        renderCanvasToTarget(modalCanvas, this.rtCompareState);
                    };

                    modalStage.addEventListener("wheel", (e) => {
                        handleWheelZoom(e, modalCanvas, this.rtCompareState, () => {
                            redrawModal();
                            renderCanvas();
                        });
                    }, { passive: false });

                    modalStage.addEventListener("dblclick", (e) => {
                        e.stopPropagation();
                        this.rtCompareState.zoom = 1.0;
                        this.rtCompareState.panX = 0;
                        this.rtCompareState.panY = 0;
                        redrawModal();
                        renderCanvas();
                    });

                    let mIsDragging = false;
                    let mIsPanning = false;
                    let mStartX = 0;
                    let mStartY = 0;
                    let mStartPanX = 0;
                    let mStartPanY = 0;

                    modalStage.addEventListener("pointerdown", (e) => {
                        const state = this.rtCompareState;
                        if (!state.isLoaded) return;

                        if (e.button === 1 || e.button === 2 || (e.shiftKey && e.button === 0) || (state.zoom > 1.01 && state.mode !== "slide" && state.mode !== "fade" && e.button === 0)) {
                            mIsPanning = true;
                            const rect = modalCanvas.getBoundingClientRect();
                            const scaleX = modalCanvas.width / rect.width;
                            const scaleY = modalCanvas.height / rect.height;
                            mStartX = e.clientX * scaleX;
                            mStartY = e.clientY * scaleY;
                            mStartPanX = state.panX || 0;
                            mStartPanY = state.panY || 0;
                            modalStage.style.cursor = "grabbing";
                            try { e.target.setPointerCapture(e.pointerId); } catch (_) {}
                            return;
                        }

                        if (e.button === 0) {
                            if (state.mode === "click") {
                                state.activeLayer = state.activeLayer === "a" ? "b" : "a";
                                redrawModal();
                                renderCanvas();
                                return;
                            }

                            mIsDragging = true;
                            try { e.target.setPointerCapture(e.pointerId); } catch (_) {}
                            handlePointerPos(e.clientX, e.clientY, modalCanvas, state, () => {
                                redrawModal();
                                renderCanvas();
                            });
                        }
                    });

                    window.addEventListener("pointermove", (e) => {
                        const state = this.rtCompareState;
                        if (mIsPanning) {
                            const rect = modalCanvas.getBoundingClientRect();
                            const scaleX = modalCanvas.width / rect.width;
                            const scaleY = modalCanvas.height / rect.height;
                            const curX = e.clientX * scaleX;
                            const curY = e.clientY * scaleY;
                            state.panX = mStartPanX + (curX - mStartX);
                            state.panY = mStartPanY + (curY - mStartY);
                            redrawModal();
                            renderCanvas();
                            return;
                        }

                        if (mIsDragging) {
                            handlePointerPos(e.clientX, e.clientY, modalCanvas, state, () => {
                                redrawModal();
                                renderCanvas();
                            });
                        }
                    });

                    window.addEventListener("pointerup", (e) => {
                        if (mIsPanning) {
                            mIsPanning = false;
                            modalStage.style.cursor = this.rtCompareState.zoom > 1.01 ? "grab" : "ew-resize";
                        }
                        if (mIsDragging) {
                            mIsDragging = false;
                        }
                        try { e.target.releasePointerCapture(e.pointerId); } catch (_) {}
                    });

                    modalStage.addEventListener("pointermove", (e) => {
                        const state = this.rtCompareState;
                        if (!mIsPanning && !mIsDragging && (state.mode === "slide" || state.mode === "fade")) {
                            handlePointerPos(e.clientX, e.clientY, modalCanvas, state, () => {
                                redrawModal();
                                renderCanvas();
                            });
                        }
                    });

                    const modalResizeObserver = new ResizeObserver(() => redrawModal());
                    modalResizeObserver.observe(modalStage);
                    requestAnimationFrame(() => redrawModal());
                };

                // External Update Hook
                this.updateImages = (dataA, dataB, mode, direction) => {
                    this.rtCompareState.imagesA = dataA || [];
                    this.rtCompareState.imagesB = dataB || [];

                    // Default mode is Slide (safely handles strings, arrays, objects)
                    if (mode) {
                        let modeStr = "";
                        if (Array.isArray(mode)) {
                            modeStr = (mode.length > 1 && mode[0].length === 1) ? mode.join("") : String(mode[0] || "");
                        } else {
                            modeStr = typeof mode === "string" ? mode : String(mode || "");
                        }
                        const modeMap = {
                            "Slide (Wipe)": "slide",
                            "Difference": "diff",
                            "Click (Toggle)": "click",
                            "Side-by-Side": "side",
                            "Fade (Dissolve)": "fade",
                            "slide": "slide",
                            "diff": "diff",
                            "click": "click",
                            "side": "side",
                            "fade": "fade",
                        };
                        this.rtCompareState.mode = modeMap[modeStr] || "slide";
                    } else {
                        this.rtCompareState.mode = "slide";
                    }

                    // Safe direction parser (safely handles non-string values)
                    if (direction) {
                        let dirStr = "";
                        if (Array.isArray(direction)) {
                            dirStr = (direction.length > 1 && direction[0].length === 1) ? direction.join("") : String(direction[0] || "");
                        } else {
                            dirStr = typeof direction === "string" ? direction : String(direction || "");
                        }
                        this.rtCompareState.direction = dirStr.toLowerCase().includes("vert") ? "vertical" : "horizontal";
                    } else {
                        this.rtCompareState.direction = "horizontal";
                    }

                    updateToolbarButtons();
                    loadCurrentImages();
                    this.setDirtyCanvas(true, true);
                };
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                const data = (message && message.images_a) ? message : (message && message.ui ? message.ui : null);
                if (data && data.images_a && data.images_b && this.updateImages) {
                    this.properties = this.properties || {};
                    this.properties.cached_images_a = data.images_a;
                    this.properties.cached_images_b = data.images_b;
                    this.properties.cached_mode = data.mode;
                    this.properties.cached_dir = data.direction;
                    this.updateImages(data.images_a, data.images_b, data.mode, data.direction);
                }
            };
        }
    },
    async setup(app) {
        api.addEventListener("executed", ({ detail }) => {
            if (!detail || !detail.node || !detail.output) return;
            const node = app.graph?.getNodeById(Number(detail.node)) || app.graph?.getNodeById(detail.node);
            if (node && (node.type === "RT_Image_Compare" || node.type === "RTImageCompare" || node.comfyClass === "RT_Image_Compare" || node.comfyClass === "RTImageCompare")) {
                const out = detail.output;
                const data = (out && out.images_a) ? out : (out && out.ui ? out.ui : null);
                if (data && data.images_a && data.images_b && node.updateImages) {
                    node.properties = node.properties || {};
                    node.properties.cached_images_a = data.images_a;
                    node.properties.cached_images_b = data.images_b;
                    node.properties.cached_mode = data.mode;
                    node.properties.cached_dir = data.direction;
                    node.updateImages(data.images_a, data.images_b, data.mode, data.direction);
                }
            }
        });
    }
});
