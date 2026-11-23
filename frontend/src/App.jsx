const { useState, useRef } = React;
const API = "http://localhost:8000";

function App() {
    const [jobId, setJobId] = useState(null);
    const [fileCount, setFileCount] = useState(0);
    const [backend, setBackend] = useState("local");
    const [threshold, setThreshold] = useState(0.91);
    const [progress, setProgress] = useState(null);
    const [clusters, setClusters] = useState(null);
    const [status, setStatus] = useState("idle");
    const fileInputRef = useRef(null);

    async function handleUpload(selectedFiles) {
        const formData = new FormData();
        for (const f of selectedFiles) formData.append("files", f);
        const resp = await fetch(`${API}/api/upload`, { method: "POST", body: formData });
        const data = await resp.json();
        setJobId(data.job_id);
        setFileCount(data.files_saved);
        setStatus("uploaded");
    }

    async function startClustering() {
        if (!jobId) return;
        setStatus("processing");
        setClusters(null);
        await fetch(`${API}/api/cluster/${jobId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ backend, threshold }),
        });
        const evtSource = new EventSource(`${API}/api/progress/${jobId}`);
        evtSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            setProgress(data);
            if (data.status === "completed") { evtSource.close(); fetchResults(); }
            else if (data.status === "failed") { evtSource.close(); setStatus("failed"); }
        };
    }

    async function fetchResults() {
        const resp = await fetch(`${API}/api/results/${jobId}`);
        const data = await resp.json();
        setClusters(data.clusters);
        setStatus("completed");
    }

    async function handleRecluster(newThreshold) {
        setThreshold(newThreshold);
        const resp = await fetch(`${API}/api/recluster`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job_id: jobId, threshold: newThreshold }),
        });
        const data = await resp.json();
        setClusters(data.clusters);
    }

    function handleDrop(e) {
        e.preventDefault();
        const dropped = Array.from(e.dataTransfer.files);
        if (dropped.length) handleUpload(dropped);
    }

    const pct = progress?.total ? Math.round((progress.done / progress.total) * 100) : 0;

    return (
        <div className="max-w-5xl mx-auto px-6 py-10">
            {/* Nav */}
            <nav className="fixed top-0 left-0 right-0 z-50 nav-blur border-b border-white/5">
                <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className="text-accent font-semibold text-sm tracking-wide">ImageCluster</span>
                        <span className="text-[10px] text-slate-500 font-mono bg-white/5 px-2 py-0.5 rounded">v2.0</span>
                    </div>
                    <span className="text-[11px] text-slate-500">Visual similarity detection at scale</span>
                </div>
            </nav>

            <div className="pt-20">
                {/* Upload */}
                {status === "idle" && (
                    <div className="fade-in">
                        <div className="text-center mb-10">
                            <h1 className="text-3xl font-bold mb-2">Find duplicate images</h1>
                            <p className="text-slate-400 text-sm max-w-md mx-auto">Upload product images or a CSV with image URLs. The system groups visually similar items into clusters using deep learning embeddings.</p>
                        </div>
                        <div
                            onDrop={handleDrop}
                            onDragOver={(e) => e.preventDefault()}
                            onClick={() => fileInputRef.current?.click()}
                            className="card glow-border glow-border-hover p-20 text-center cursor-pointer transition-all duration-300"
                        >
                            <div className="text-accent text-3xl mb-3">⬆</div>
                            <p className="text-slate-300 text-sm font-medium">Drop images or CSV here</p>
                            <p className="text-slate-500 text-xs mt-2">JPG, PNG, or CSV with image_url column</p>
                            <input ref={fileInputRef} type="file" multiple accept=".jpg,.jpeg,.png,.csv" className="hidden" onChange={(e) => handleUpload(Array.from(e.target.files))} />
                        </div>
                    </div>
                )}

                {/* Settings */}
                {status === "uploaded" && (
                    <div className="fade-in card glow-border p-8 space-y-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <h2 className="text-lg font-semibold">Ready to cluster</h2>
                                <p className="text-xs text-slate-500 mt-1">{fileCount} file(s) uploaded</p>
                            </div>
                            <span className="text-[10px] font-mono text-accent bg-accent/10 px-3 py-1 rounded-full">JOB {jobId}</span>
                        </div>

                        <div>
                            <label className="text-[11px] uppercase tracking-wider text-slate-500 block mb-3">Embedding Backend</label>
                            <div className="flex gap-3">
                                {[
                                    { id: "local", label: "Local", desc: "EfficientNet-B0" },
                                    { id: "titan", label: "Titan", desc: "AWS Bedrock" },
                                    { id: "vertex", label: "Vertex", desc: "Google Cloud" },
                                ].map((b) => (
                                    <button
                                        key={b.id}
                                        onClick={() => setBackend(b.id)}
                                        className={`flex-1 p-3 rounded-xl border text-center transition-all duration-200 ${
                                            backend === b.id
                                                ? "border-accent/40 bg-accent/5 shadow-[0_0_15px_-3px_rgba(0,212,170,0.2)]"
                                                : "border-white/6 bg-white/2 hover:border-white/12"
                                        }`}
                                    >
                                        <div className={`text-sm font-medium ${backend === b.id ? "text-accent" : "text-slate-300"}`}>{b.label}</div>
                                        <div className="text-[10px] text-slate-500 mt-0.5">{b.desc}</div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <label className="text-[11px] uppercase tracking-wider text-slate-500">Similarity Threshold</label>
                                <span className="text-sm font-mono text-accent">{threshold.toFixed(2)}</span>
                            </div>
                            <input type="range" min="0.70" max="0.99" step="0.01" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-full h-1.5 rounded-full appearance-none cursor-pointer" />
                            <div className="flex justify-between text-[10px] text-slate-600 mt-1">
                                <span>loose (more clusters)</span>
                                <span>strict (fewer clusters)</span>
                            </div>
                        </div>

                        <button onClick={startClustering} className="w-full bg-accent hover:bg-accent/90 text-surface font-semibold text-sm py-3 rounded-xl transition-all duration-200 shadow-[0_0_20px_-5px_rgba(0,212,170,0.3)]">
                            Start Clustering
                        </button>
                    </div>
                )}

                {/* Progress */}
                {status === "processing" && (
                    <div className="fade-in card glow-border p-8">
                        <div className="flex justify-between items-center mb-4">
                            <span className="text-sm font-medium text-slate-300">Processing images...</span>
                            <span className="text-xs font-mono text-accent">{progress?.done || 0}/{progress?.total || "?"}</span>
                        </div>
                        <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                            <div className="bg-accent h-2 rounded-full transition-all duration-500 progress-glow" style={{ width: `${pct}%` }} />
                        </div>
                        <p className="text-[11px] text-slate-500 mt-3">Embedding with {backend === "local" ? "EfficientNet-B0" : backend === "titan" ? "Amazon Titan" : "Vertex AI"} → FAISS similarity search → Union-Find clustering</p>
                    </div>
                )}

                {/* Results */}
                {status === "completed" && clusters && (
                    <div className="fade-in space-y-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <h2 className="text-lg font-semibold">{clusters.length} clusters found</h2>
                                <p className="text-xs text-slate-500">{clusters.reduce((s, c) => s + c.size, 0)} items grouped</p>
                            </div>
                            <div className="flex items-center gap-3 card px-4 py-2">
                                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Threshold</span>
                                <input type="range" min="0.70" max="0.99" step="0.01" value={threshold} onChange={(e) => handleRecluster(parseFloat(e.target.value))} className="w-32 h-1 rounded appearance-none cursor-pointer" />
                                <span className="text-xs font-mono text-accent w-8">{threshold.toFixed(2)}</span>
                            </div>
                        </div>

                        {clusters.map((cluster) => (
                            <div key={cluster.cluster_id} className="card glow-border-hover p-5 transition-all duration-300">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <span className="text-[10px] font-mono text-accent bg-accent/10 px-2 py-0.5 rounded">CLUSTER {cluster.cluster_id + 1}</span>
                                        <span className="text-xs text-slate-400">{cluster.size} items</span>
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-3">
                                    {cluster.members.slice(0, 8).map((member) => (
                                        <div key={member.id} className="group">
                                            {member.filename ? (
                                                <img
                                                    src={`${API}/api/images/${jobId}/${member.filename}`}
                                                    alt={member.id}
                                                    className="w-28 h-28 object-cover rounded-xl border border-white/8 group-hover:border-accent/30 transition-all duration-200"
                                                />
                                            ) : (
                                                <div className="w-28 h-28 bg-white/3 rounded-xl border border-white/6 flex items-center justify-center">
                                                    <span className="text-[10px] text-slate-600">No image</span>
                                                </div>
                                            )}
                                            <p className="text-[9px] font-mono text-slate-500 mt-1.5 text-center truncate w-28">{member.id}</p>
                                        </div>
                                    ))}
                                    {cluster.size > 8 && (
                                        <div className="w-28 h-28 bg-white/3 rounded-xl border border-white/6 flex items-center justify-center">
                                            <span className="text-xs text-slate-500">+{cluster.size - 8}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Error */}
                {status === "failed" && (
                    <div className="fade-in card border-red-900/30 bg-red-950/20 p-6">
                        <p className="text-sm text-red-300">Clustering failed. Check backend logs for details.</p>
                    </div>
                )}
            </div>
        </div>
    );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
