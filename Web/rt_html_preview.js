import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "RareTutor.HtmlPreviewNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RTLTX2HtmlPreview") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const textWidget = this.widgets.find(w => w.name === "html_text");
                if (!textWidget) return;

                // 1. SAFELY HIDE THE NATIVE WIDGET 
                // (Keep its data type intact so your workflow saves correctly)
                if (textWidget.element) {
                    textWidget.element.style.display = "none";
                }
                textWidget.computeSize = () => [0, 0]; 
                textWidget.draw = () => {}; 

                // 2. CREATE UNIFIED CONTAINER
                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%",
                    height: "100%",
                    position: "relative",
                    minHeight: "150px"
                });

                // 3. CREATE THE CODE EDITOR
                const editor = document.createElement("textarea");
                // THE MTB SECRET: Add ComfyUI's native class so it respects the input!
                editor.className = "comfy-multiline-input"; 
                Object.assign(editor.style, {
                    width: "100%",
                    height: "100%",
                    boxSizing: "border-box",
                    backgroundColor: "var(--comfy-input-bg, #1e1e1e)",
                    color: "var(--input-text, #cccccc)",
                    border: "1px solid var(--border-color, #444)",
                    borderRadius: "6px",
                    padding: "10px",
                    fontFamily: "monospace",
                    fontSize: "14px",
                    resize: "none",
                    position: "absolute",
                    top: "0",
                    left: "0",
                    display: "none", // Hidden by default
                    zIndex: 2,
                    // FORCE browser to allow typing and clicking
                    userSelect: "text",
                    cursor: "text",
                    pointerEvents: "auto"
                });
                editor.value = textWidget.value;

                // 4. CREATE THE HTML PREVIEW
                const preview = document.createElement("div");
                Object.assign(preview.style, {
                    width: "100%",
                    height: "100%",
                    boxSizing: "border-box",
                    backgroundColor: "var(--comfy-input-bg, #2b2b2b)",
                    color: "var(--input-text, #f0f0f0)",
                    border: "1px solid var(--border-color, #444)",
                    borderRadius: "6px",
                    padding: "10px",
                    overflow: "auto",
                    fontFamily: "Arial, sans-serif",
                    position: "absolute",
                    top: "0",
                    left: "0",
                    zIndex: 1,
                    // FORCE browser to allow clicking the preview
                    userSelect: "text",
                    cursor: "pointer",
                    pointerEvents: "auto"
                });
                preview.innerHTML = textWidget.value;

                container.appendChild(editor);
                container.appendChild(preview);

                this.addDOMWidget("HTML_Unified", "div", container);

                // --- SHIELD EVENTS FROM LITEGRAPH ---
                const preventDrag = (e) => e.stopPropagation();
                container.addEventListener("mousedown", preventDrag);
                container.addEventListener("pointerdown", preventDrag);
                container.addEventListener("wheel", preventDrag);
                container.addEventListener("keydown", preventDrag);

                // --- TOGGLE LOGIC ---
                
                // Click Preview -> Show Editor & Focus
                preview.addEventListener("click", () => {
                    preview.style.display = "none";
                    editor.style.display = "block";
                    editor.focus();
                });

                // Leave Editor -> Save & Show Preview
                editor.addEventListener("blur", () => {
                    textWidget.value = editor.value; // Save to backend
                    preview.innerHTML = editor.value; // Update HTML
                    editor.style.display = "none";
                    preview.style.display = "block";
                });

// Sync data if the workflow is loaded from a JSON file
                const originalCallback = textWidget.callback;
                textWidget.callback = function(value) {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    editor.value = value;
                    preview.innerHTML = value;
                };

                // --- LISTEN FOR INCOMING WIRE DATA ---
                // This catches the {"ui": {"text": [...]}} sent from Python
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    if (onExecuted) onExecuted.apply(this, arguments);
                    if (message && message.text) {
                        const newHtml = message.text[0];
                        textWidget.value = newHtml; // Update hidden node data
                        editor.value = newHtml;     // Update code editor
                        preview.innerHTML = newHtml;// Render new HTML visually
                    }
                };

            }; // <--- End of onNodeCreated
        }
    }
});