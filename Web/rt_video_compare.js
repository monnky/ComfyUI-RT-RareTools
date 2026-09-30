import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "RareTutor.VideoCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RT_Video_Compare" || nodeData.name === "RTVideoCompare") {
            // Style helper for node colors
            const applyNodeStyle = (node) => {
                node.color = "#004322"; // header color
                node.bgcolor = "#002211"; // body color
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) onConfigure.apply(this, arguments);
                applyNodeStyle(this);
                setTimeout(() => applyNodeStyle(this), 50);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                applyNodeStyle(this);
                setTimeout(() => applyNodeStyle(this), 50);

                // Set generous default size for video inspection
                if (!this.size || this.size[0] < 500 || this.size[1] < 460) {
                    this.size = [520, 480];
                }

                this.rtVideoState = {
                    mode: "slide", // "slide", "click", "side", "diff", "fade"
                    splitPct: 50,
                    activeLayer: "a",
                    isPlaying: true,
                    fps: 24,
                    frameCount: 0,
                    swapped: false,
                    speed: 1.0,
                    loop: true,
                    videoAData: null,
                    videoBData: null,
                };

                // Master Container
                const container = document.createElement("div");
                container.className = "rt-video-compare-container";
                Object.assign(container.style, {
                    width: "100%",
                    height: "100%",
                    minHeight: "340px",
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
                container.addEventListener("wheel", stopEvent);

                // 1. Video Viewer Stage
                const stage = document.createElement("div");
                Object.assign(stage.style, {
                    position: "relative",
                    flex: "1 1 auto",
                    width: "100%",
                    height: "100%",
                    overflow: "hidden",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "ew-resize",
                    backgroundColor: "#0d0d0f",
                });

                // Empty Placeholder
                const emptyPlaceholder = document.createElement("div");
                emptyPlaceholder.innerText = "Connect Video A & B and Run";
                Object.assign(emptyPlaceholder.style, {
                    color: "rgba(255, 255, 255, 0.35)",
                    fontSize: "14px",
                    fontWeight: "500",
                    letterSpacing: "0.5px",
                    textAlign: "center",
                    pointerEvents: "none",
                });
                stage.appendChild(emptyPlaceholder);

                // Video Box Wrapper
                const vidBox = document.createElement("div");
                Object.assign(vidBox.style, {
                    position: "relative",
                    width: "100%",
                    height: "100%",
                    display: "none",
                    alignItems: "center",
                    justifyContent: "center",
                });
                stage.appendChild(vidBox);

                // Video B (Underneath / Base Layer)
                const vidB = document.createElement("video");
                vidB.muted = true;
                vidB.playsInline = true;
                vidB.loop = true;
                vidB.autoplay = true;
                Object.assign(vidB.style, {
                    position: "absolute",
                    top: "0",
                    left: "0",
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    pointerEvents: "none",
                });
                vidBox.appendChild(vidB);

                // Video A (Top Layer / Clipped)
                const vidA = document.createElement("video");
                vidA.muted = true;
                vidA.playsInline = true;
                vidA.loop = true;
                vidA.autoplay = true;
                Object.assign(vidA.style, {
                    position: "absolute",
                    top: "0",
                    left: "0",
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    pointerEvents: "none",
                    clipPath: "inset(0 50% 0 0)",
                });
                vidBox.appendChild(vidA);

                // Divider Line
                const divider = document.createElement("div");
                Object.assign(divider.style, {
                    position: "absolute",
                    top: "0",
                    bottom: "0",
                    left: "50%",
                    width: "2px",
                    backgroundColor: "#ffffff",
                    boxShadow: "0 0 8px rgba(0, 0, 0, 0.8), 0 0 4px rgba(255, 255, 255, 0.8)",
                    transform: "translateX(-50%)",
                    pointerEvents: "none",
                    zIndex: "10",
                });
                const dividerKnob = document.createElement("div");
                dividerKnob.innerHTML = "&lt;&gt;";
                Object.assign(dividerKnob.style, {
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                    width: "30px",
                    height: "30px",
                    borderRadius: "50%",
                    backgroundColor: "#202026",
                    color: "#ffffff",
                    border: "2px solid #ffffff",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "11px",
                    fontWeight: "bold",
                    boxShadow: "0 2px 10px rgba(0,0,0,0.6)",
                });
                divider.appendChild(dividerKnob);
                vidBox.appendChild(divider);

                // Badges
                const badgeA = document.createElement("div");
                badgeA.innerText = "A";
                Object.assign(badgeA.style, {
                    position: "absolute",
                    top: "10px",
                    left: "10px",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    backgroundColor: "rgba(10, 10, 14, 0.75)",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                    color: "#00e5ff",
                    fontSize: "11px",
                    fontWeight: "700",
                    zIndex: "15",
                    backdropFilter: "blur(4px)",
                    pointerEvents: "none",
                });
                vidBox.appendChild(badgeA);

                const badgeB = document.createElement("div");
                badgeB.innerText = "B";
                Object.assign(badgeB.style, {
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    backgroundColor: "rgba(10, 10, 14, 0.75)",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                    color: "#ff007f",
                    fontSize: "11px",
                    fontWeight: "700",
                    zIndex: "15",
                    backdropFilter: "blur(4px)",
                    pointerEvents: "none",
                });
                vidBox.appendChild(badgeB);

                // --- 2. Synchronized Video Transport Engine ---
                const syncVideos = () => {
                    vidA.onplay = () => vidB.play();
                    vidA.onpause = () => vidB.pause();
                    vidA.onseeking = () => { vidB.currentTime = vidA.currentTime; };
                    vidA.onseeked = () => { vidB.currentTime = vidA.currentTime; };

                    vidA.ontimeupdate = () => {
                        // Drift correction: keep B within 0.04s of A
                        if (Math.abs(vidA.currentTime - vidB.currentTime) > 0.04) {
                            vidB.currentTime = vidA.currentTime;
                        }
                        updateTimelineProgress();
                    };
                };
                syncVideos();

                // 3. Timeline Scrubber Bar
                const timelineBar = document.createElement("div");
                Object.assign(timelineBar.style, {
                    height: "26px",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "0 10px",
                    backgroundColor: "rgba(0, 34, 17, 0.95)",
                    borderTop: "1px solid rgba(0, 67, 34, 0.5)",
                    zIndex: "20",
                });

                const scrubber = document.createElement("input");
                scrubber.type = "range";
                scrubber.min = "0";
                scrubber.max = "100";
                scrubber.value = "0";
                scrubber.step = "0.1";
                Object.assign(scrubber.style, {
                    flex: "1",
                    cursor: "pointer",
                    accentColor: "#3182ce",
                    height: "4px",
                });

                scrubber.addEventListener("input", () => {
                    if (vidA.duration) {
                        const targetTime = (parseFloat(scrubber.value) / 100) * vidA.duration;
                        vidA.currentTime = targetTime;
                        vidB.currentTime = targetTime;
                    }
                });

                const timeLabel = document.createElement("span");
                timeLabel.innerText = "0 / 0 | 00:00";
                Object.assign(timeLabel.style, {
                    fontSize: "11px",
                    fontFamily: "monospace",
                    color: "#81e6d9",
                    whiteSpace: "nowrap",
                });

                const updateTimelineProgress = () => {
                    if (!vidA.duration) return;
                    const pct = (vidA.currentTime / vidA.duration) * 100;
                    scrubber.value = pct;
                    const currentFrame = Math.floor(vidA.currentTime * this.rtVideoState.fps);
                    const totalFrames = Math.max(1, Math.floor(vidA.duration * this.rtVideoState.fps));
                    const curSec = vidA.currentTime.toFixed(2);
                    timeLabel.innerText = `${currentFrame}/${totalFrames} | ${curSec}s`;
                };

                timelineBar.appendChild(scrubber);
                timelineBar.appendChild(timeLabel);

                // 4. Transport Control Bar (Play, Pause, Step, Speed, Fullscreen)
                const controlBar = document.createElement("div");
                Object.assign(controlBar.style, {
                    height: "36px",
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
                    { id: "click", label: "Click" },
                    { id: "side", label: "Side" },
                    { id: "diff", label: "Diff" },
                    { id: "fade", label: "Fade" },
                ];

                const modeButtons = {};
                modes.forEach(m => {
                    const btn = document.createElement("button");
                    btn.innerText = m.label;
                    Object.assign(btn.style, {
                        backgroundColor: m.id === "slide" ? "#2b6cb0" : "transparent",
                        color: m.id === "slide" ? "#ffffff" : "#a0aec0",
                        border: "none",
                        borderRadius: "4px",
                        padding: "3px 8px",
                        fontSize: "11px",
                        fontWeight: "600",
                        cursor: "pointer",
                        outline: "none",
                    });

                    btn.addEventListener("click", () => {
                        this.rtVideoState.mode = m.id;
                        updateModeUI();
                    });
                    modeGroup.appendChild(btn);
                    modeButtons[m.id] = btn;
                });
                controlBar.appendChild(modeGroup);

                // Playback Actions (Play/Pause, Step Back, Step Forward, Speed)
                const playbackGroup = document.createElement("div");
                Object.assign(playbackGroup.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "5px",
                });

                const playBtn = document.createElement("button");
                playBtn.innerText = "|| Pause";
                Object.assign(playBtn.style, {
                    backgroundColor: "rgba(255, 255, 255, 0.08)",
                    color: "#ffffff",
                    border: "none",
                    borderRadius: "4px",
                    padding: "3px 8px",
                    fontSize: "11px",
                    fontWeight: "600",
                    cursor: "pointer",
                });
                playBtn.addEventListener("click", () => {
                    if (vidA.paused) {
                        vidA.play();
                        vidB.play();
                        playBtn.innerText = "|| Pause";
                    } else {
                        vidA.pause();
                        vidB.pause();
                        playBtn.innerText = "> Play";
                    }
                });

                // Frame Step Buttons
                const stepBackBtn = document.createElement("button");
                stepBackBtn.innerText = "< Step";
                const stepFwdBtn = document.createElement("button");
                stepFwdBtn.innerText = "Step >";
                [stepBackBtn, stepFwdBtn].forEach(b => {
                    Object.assign(b.style, {
                        backgroundColor: "rgba(255, 255, 255, 0.06)",
                        color: "#cbd5e0",
                        border: "none",
                        borderRadius: "3px",
                        padding: "3px 6px",
                        fontSize: "10px",
                        cursor: "pointer",
                    });
                });

                stepBackBtn.addEventListener("click", () => {
                    vidA.pause();
                    vidB.pause();
                    playBtn.innerText = "> Play";
                    const frameTime = 1.0 / this.rtVideoState.fps;
                    vidA.currentTime = Math.max(0, vidA.currentTime - frameTime);
                    vidB.currentTime = vidA.currentTime;
                });

                stepFwdBtn.addEventListener("click", () => {
                    vidA.pause();
                    vidB.pause();
                    playBtn.innerText = "> Play";
                    const frameTime = 1.0 / this.rtVideoState.fps;
                    vidA.currentTime = Math.min(vidA.duration || 999, vidA.currentTime + frameTime);
                    vidB.currentTime = vidA.currentTime;
                });

                // Speed Selector
                const speedSelect = document.createElement("select");
                ["0.25", "0.5", "1.0", "2.0"].forEach(s => {
                    const opt = document.createElement("option");
                    opt.value = s;
                    opt.innerText = `${s}x`;
                    if (s === "1.0") opt.selected = true;
                    speedSelect.appendChild(opt);
                });
                Object.assign(speedSelect.style, {
                    backgroundColor: "rgba(255, 255, 255, 0.08)",
                    color: "#cbd5e0",
                    border: "none",
                    borderRadius: "4px",
                    padding: "2px 4px",
                    fontSize: "11px",
                    cursor: "pointer",
                });
                speedSelect.addEventListener("change", () => {
                    const spd = parseFloat(speedSelect.value);
                    vidA.playbackRate = spd;
                    vidB.playbackRate = spd;
                });

                // Swap A <-> B Button
                const swapBtn = document.createElement("button");
                swapBtn.innerText = "Swap";
                Object.assign(swapBtn.style, {
                    backgroundColor: "rgba(255, 255, 255, 0.06)",
                    color: "#cbd5e0",
                    border: "none",
                    borderRadius: "4px",
                    padding: "3px 6px",
                    fontSize: "11px",
                    cursor: "pointer",
                });
                swapBtn.addEventListener("click", () => {
                    this.rtVideoState.swapped = !this.rtVideoState.swapped;
                    renderActiveVideos();
                });

                // Fullscreen Modal Button
                const expandBtn = document.createElement("button");
                expandBtn.innerText = "Zoom";
                Object.assign(expandBtn.style, {
                    backgroundColor: "rgba(255, 255, 255, 0.08)",
                    color: "#ffffff",
                    border: "1px solid rgba(255, 255, 255, 0.15)",
                    borderRadius: "4px",
                    padding: "3px 8px",
                    fontSize: "11px",
                    fontWeight: "600",
                    cursor: "pointer",
                });
                expandBtn.addEventListener("click", () => openFullscreenModal());

                playbackGroup.appendChild(playBtn);
                playbackGroup.appendChild(stepBackBtn);
                playbackGroup.appendChild(stepFwdBtn);
                playbackGroup.appendChild(speedSelect);
                playbackGroup.appendChild(swapBtn);
                playbackGroup.appendChild(expandBtn);

                controlBar.appendChild(playbackGroup);

                container.appendChild(stage);
                container.appendChild(timelineBar);
                container.appendChild(controlBar);

                this.addDOMWidget("rt_video_compare_widget", "div", container, { serialize: false });

                // --- 5. Interactive Wipe Slider Logic ---
                let isDragging = false;

                const updateSplit = (clientX) => {
                    const rect = stage.getBoundingClientRect();
                    let pct = ((clientX - rect.left) / rect.width) * 100;
                    pct = Math.max(0, Math.min(100, pct));
                    this.rtVideoState.splitPct = pct;

                    if (this.rtVideoState.mode === "slide") {
                        vidA.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
                        divider.style.left = `${pct}%`;
                    } else if (this.rtVideoState.mode === "fade") {
                        vidA.style.opacity = (100 - pct) / 100;
                    }
                };

                stage.addEventListener("pointerdown", (e) => {
                    if (this.rtVideoState.mode === "click") {
                        this.rtVideoState.activeLayer = this.rtVideoState.activeLayer === "a" ? "b" : "a";
                        applyClickMode();
                        return;
                    }
                    isDragging = true;
                    updateSplit(e.clientX);
                });

                window.addEventListener("pointermove", (e) => {
                    if (!isDragging) return;
                    updateSplit(e.clientX);
                });

                window.addEventListener("pointerup", () => {
                    isDragging = false;
                });

                stage.addEventListener("pointermove", (e) => {
                    if (this.rtVideoState.mode === "slide" && !isDragging) {
                        updateSplit(e.clientX);
                    }
                });

                const applyClickMode = () => {
                    if (this.rtVideoState.activeLayer === "a") {
                        vidA.style.opacity = "1";
                        badgeA.style.display = "block";
                        badgeB.style.display = "none";
                    } else {
                        vidA.style.opacity = "0";
                        badgeA.style.display = "none";
                        badgeB.style.display = "block";
                    }
                };

                const updateModeUI = () => {
                    const m = this.rtVideoState.mode;

                    Object.keys(modeButtons).forEach(k => {
                        const active = k === m;
                        modeButtons[k].style.backgroundColor = active ? "#2b6cb0" : "transparent";
                        modeButtons[k].style.color = active ? "#ffffff" : "#a0aec0";
                    });

                    vidA.style.display = "block";
                    vidB.style.display = "block";
                    vidA.style.opacity = "1";
                    vidB.style.opacity = "1";
                    vidA.style.mixBlendMode = "normal";
                    vidA.style.clipPath = "none";
                    divider.style.display = "none";
                    vidBox.style.flexDirection = "row";
                    vidA.style.position = "absolute";
                    vidB.style.position = "absolute";
                    vidA.style.width = "100%";
                    vidB.style.width = "100%";
                    vidA.style.height = "100%";
                    vidB.style.height = "100%";
                    vidA.style.left = "0";
                    vidB.style.left = "0";
                    vidA.style.top = "0";
                    vidB.style.top = "0";
                    stage.style.cursor = "default";
                    badgeA.style.display = "block";
                    badgeB.style.display = "block";

                    if (m === "slide") {
                        divider.style.display = "block";
                        stage.style.cursor = "ew-resize";
                        updateSplit(stage.getBoundingClientRect().left + stage.clientWidth * (this.rtVideoState.splitPct / 100));
                    } else if (m === "click") {
                        stage.style.cursor = "pointer";
                        applyClickMode();
                    } else if (m === "side") {
                        vidBox.style.display = "flex";
                        vidBox.style.flexDirection = "row";
                        vidA.style.position = "relative";
                        vidB.style.position = "relative";
                        vidA.style.width = "50%";
                        vidB.style.width = "50%";
                    } else if (m === "diff") {
                        vidA.style.mixBlendMode = "difference";
                        vidA.style.clipPath = "none";
                    } else if (m === "fade") {
                        stage.style.cursor = "ew-resize";
                        vidA.style.opacity = (100 - this.rtVideoState.splitPct) / 100;
                    }
                };

                const renderActiveVideos = () => {
                    const state = this.rtVideoState;
                    const dataA = state.swapped ? state.videoBData : state.videoAData;
                    const dataB = state.swapped ? state.videoAData : state.videoBData;

                    if (!dataA || !dataB) {
                        emptyPlaceholder.style.display = "block";
                        vidBox.style.display = "none";
                        return;
                    }

                    emptyPlaceholder.style.display = "none";
                    vidBox.style.display = "flex";

                    const urlA = api.apiURL(`/view?filename=${encodeURIComponent(dataA.filename)}&type=${dataA.type}&subfolder=${dataA.subfolder || ""}`);
                    const urlB = api.apiURL(`/view?filename=${encodeURIComponent(dataB.filename)}&type=${dataB.type}&subfolder=${dataB.subfolder || ""}`);

                    vidA.src = urlA;
                    vidB.src = urlB;
                    vidA.load();
                    vidB.load();
                    vidA.play().catch(() => {});
                    vidB.play().catch(() => {});

                    const labelTextA = state.swapped ? "B" : "A";
                    const labelTextB = state.swapped ? "A" : "B";
                    badgeA.innerText = `${labelTextA} (${dataA.width || ""}x${dataA.height || ""})`;
                    badgeB.innerText = `${labelTextB} (${dataB.width || ""}x${dataB.height || ""})`;

                    updateModeUI();
                };

                // --- 6. Fullscreen Modal Lightbox ---
                const openFullscreenModal = () => {
                    const modal = document.createElement("div");
                    Object.assign(modal.style, {
                        position: "fixed",
                        top: "0",
                        left: "0",
                        width: "100vw",
                        height: "100vh",
                        backgroundColor: "rgba(5, 5, 8, 0.95)",
                        backdropFilter: "blur(8px)",
                        zIndex: "99999",
                        display: "flex",
                        flexDirection: "column",
                    });

                    // Header
                    const modalHeader = document.createElement("div");
                    Object.assign(modalHeader.style, {
                        height: "48px",
                        padding: "0 20px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        backgroundColor: "#16161c",
                        borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
                    });
                    const modalTitle = document.createElement("div");
                    modalTitle.innerText = "RT Video Compare (Full Resolution Inspection)";
                    Object.assign(modalTitle.style, {
                        color: "#ffffff",
                        fontSize: "14px",
                        fontWeight: "600",
                    });
                    const closeBtn = document.createElement("button");
                    closeBtn.innerText = "Close";
                    Object.assign(closeBtn.style, {
                        backgroundColor: "#e53e3e",
                        color: "#ffffff",
                        border: "none",
                        borderRadius: "4px",
                        padding: "6px 14px",
                        fontSize: "12px",
                        fontWeight: "600",
                        cursor: "pointer",
                    });
                    closeBtn.addEventListener("click", () => document.body.removeChild(modal));
                    modalHeader.appendChild(modalTitle);
                    modalHeader.appendChild(closeBtn);

                    // Body
                    const modalBody = document.createElement("div");
                    Object.assign(modalBody.style, {
                        flex: "1",
                        position: "relative",
                        overflow: "hidden",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        cursor: "ew-resize",
                    });

                    const mVidB = document.createElement("video");
                    mVidB.src = vidB.src;
                    mVidB.muted = true;
                    mVidB.loop = true;
                    mVidB.autoplay = true;
                    Object.assign(mVidB.style, {
                        position: "absolute",
                        maxWidth: "96%",
                        maxHeight: "96%",
                        objectFit: "contain",
                        pointerEvents: "none",
                    });

                    const mVidA = document.createElement("video");
                    mVidA.src = vidA.src;
                    mVidA.muted = true;
                    mVidA.loop = true;
                    mVidA.autoplay = true;
                    Object.assign(mVidA.style, {
                        position: "absolute",
                        maxWidth: "96%",
                        maxHeight: "96%",
                        objectFit: "contain",
                        pointerEvents: "none",
                        clipPath: "inset(0 50% 0 0)",
                    });

                    const mDivider = document.createElement("div");
                    Object.assign(mDivider.style, {
                        position: "absolute",
                        top: "2%",
                        bottom: "2%",
                        left: "50%",
                        width: "2px",
                        backgroundColor: "#ffffff",
                        boxShadow: "0 0 10px rgba(0,0,0,0.8)",
                        transform: "translateX(-50%)",
                        pointerEvents: "none",
                    });

                    modalBody.appendChild(mVidB);
                    modalBody.appendChild(mVidA);
                    modalBody.appendChild(mDivider);

                    // Synchronize modal videos
                    mVidA.currentTime = vidA.currentTime;
                    mVidB.currentTime = vidA.currentTime;
                    mVidA.play().catch(() => {});
                    mVidB.play().catch(() => {});

                    let mDragging = false;
                    const updateModalSplit = (clientX) => {
                        const rect = modalBody.getBoundingClientRect();
                        let pct = ((clientX - rect.left) / rect.width) * 100;
                        pct = Math.max(0, Math.min(100, pct));
                        mVidA.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
                        mDivider.style.left = `${pct}%`;
                    };

                    modalBody.addEventListener("pointerdown", (e) => {
                        mDragging = true;
                        updateModalSplit(e.clientX);
                    });
                    window.addEventListener("pointermove", (e) => {
                        if (mDragging) updateModalSplit(e.clientX);
                    });
                    window.addEventListener("pointerup", () => {
                        mDragging = false;
                    });
                    modalBody.addEventListener("pointermove", (e) => {
                        if (!mDragging) updateModalSplit(e.clientX);
                    });

                    modal.appendChild(modalHeader);
                    modal.appendChild(modalBody);
                    document.body.appendChild(modal);
                };

                // Store update handle on node instance
                this.updateVideos = (dataA, dataB, fps, mode) => {
                    this.rtVideoState.videoAData = dataA;
                    this.rtVideoState.videoBData = dataB;
                    if (fps) this.rtVideoState.fps = fps;
                    if (mode) {
                        const modeMap = {
                            "Slide (Wipe)": "slide",
                            "Click (Toggle)": "click",
                            "Side-by-Side": "side",
                            "Difference": "diff",
                            "Fade (Dissolve)": "fade",
                        };
                        this.rtVideoState.mode = modeMap[mode] || "slide";
                    }
                    renderActiveVideos();
                    this.setDirtyCanvas(true, true);
                };
            };

            // Helper: ComfyUI wraps all ui values in arrays — unwrap if needed
            const unwrap = (v) => (Array.isArray(v) ? v[0] : v);

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                const data = (message && message.video_a) ? message : (message && message.ui ? message.ui : null);
                if (data && data.video_a && data.video_b && this.updateVideos) {
                    this.updateVideos(unwrap(data.video_a), unwrap(data.video_b), unwrap(data.fps), unwrap(data.mode));
                }
            };
        }
    },
    async setup(app) {
        // Helper: ComfyUI wraps all ui values in arrays — unwrap if needed
        const unwrap = (v) => (Array.isArray(v) ? v[0] : v);
        api.addEventListener("executed", ({ detail }) => {
            if (!detail || !detail.node || !detail.output) return;
            const node = app.graph?.getNodeById(Number(detail.node)) || app.graph?.getNodeById(detail.node);
            if (node && (node.type === "RT_Video_Compare" || node.type === "RTVideoCompare" || node.comfyClass === "RT_Video_Compare" || node.comfyClass === "RTVideoCompare")) {
                const out = detail.output;
                const data = (out && out.video_a) ? out : (out && out.ui ? out.ui : null);
                if (data && data.video_a && data.video_b && node.updateVideos) {
                    node.updateVideos(unwrap(data.video_a), unwrap(data.video_b), unwrap(data.fps), unwrap(data.mode));
                }
            }
        });
    }
});
