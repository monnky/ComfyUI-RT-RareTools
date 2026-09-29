import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "RareTutor.SyncTextNodes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // We only want to apply this auto-update trick to these two specific nodes
        if (nodeData.name === "RTLTX2TextInput" || nodeData.name === "RTLTX2StickyNote") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                // If Python sent us new text data during a run...
                if (message && message.text) {
                    const newText = message.text[0];

                    // Find the correct text box ('text' for Input, 'note_text' for Sticky Note)
                    const widgetName = nodeData.name === "RTLTX2TextInput" ? "text" : "note_text";
                    const widget = this.widgets?.find(w => w.name === widgetName);

                    if (widget) {
                        // Inject the new text into the box
                        widget.value = newText;
                        
                        // Force ComfyUI's canvas to redraw so you can physically see the change
                        this.setDirtyCanvas(true, true);
                    }
                }
            };
            if (nodeData.name === "RTLTX2StickyNote") {
                const applyStyle = (node) => {
                    node.color = "#004322"; // header color
                    node.bgcolor = "#002211"; // body color
                    const widget = node.widgets?.find(w => w.name === "note_text");
                    if (widget && widget.inputEl) {
                        widget.inputEl.style.setProperty("background-color", "#004322", "important");
                        widget.inputEl.style.setProperty("color", "#fde624", "important");
                        widget.inputEl.style.setProperty("font-size", "16px", "important");
                        widget.inputEl.style.setProperty("font-family", "'Calibri', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif", "important");
                        widget.inputEl.style.setProperty("font-weight", "bold", "important");
                    }
                };

                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function() {
                    if (onNodeCreated) onNodeCreated.apply(this, arguments);
                    applyStyle(this);
                    setTimeout(() => applyStyle(this), 100);
                };

                const onConfigure = nodeType.prototype.onConfigure;
                nodeType.prototype.onConfigure = function() {
                    if (onConfigure) onConfigure.apply(this, arguments);
                    applyStyle(this);
                    setTimeout(() => applyStyle(this), 100);
                };
            }
        }
    }
});