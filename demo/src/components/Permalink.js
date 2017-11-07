import React from 'react';

// from https://stackoverflow.com/a/33928558/1076346
function copyToClipboard(text) {
    if (window.clipboardData && window.clipboardData.setData) {
        // IE specific code path to prevent textarea being shown while dialog is visible.
        return window.clipboardData.setData("Text", text);

    } else if (document.queryCommandSupported && document.queryCommandSupported("copy")) {
        var textarea = document.createElement("textarea");
        textarea.textContent = text;
        textarea.style.position = "fixed";  // Prevent scrolling to bottom of page in MS Edge.
        document.body.appendChild(textarea);
        textarea.select();
        try {
            return document.execCommand("copy");  // Security exception may be thrown by some browsers.
        } catch (ex) {
            console.warn("Copy to clipboard failed.", ex);
            return false;
        } finally {
            document.body.removeChild(textarea);
        }
    }
}

class Permalink extends React.Component {


    render() {
        const { slug } = this.props;

        // If no slug, don't render anything.
        if (!slug) {
            return null;
        }

        const hashRegex = /#\/?$/;
        const demoRoot = window.location.href.replace(hashRegex, '');
        const permalink = demoRoot + 'permalink/' + slug;

        return (
            <div className="model__content">
                <div className="form__field">
                    <label>Permalink:</label>
                    <input type="text" disabled="true" className="permalink" id="permalink" value={permalink}/>
                    <button className="copy__to__clipboard" onClick={() => copyToClipboard(permalink)}>Copy to Clipboard</button>
                </div>
            </div>
        )
    }
}

export default Permalink;
