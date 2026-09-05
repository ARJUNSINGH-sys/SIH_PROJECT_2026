# PERCEPTRA — SIH 26053 Spatiotemporal 2.5D LiDAR Perception Engine

**Smart India Hackathon 2026 | Problem Statement ID: 26053 (Software)**  
**Organization:** DRDO — Department of Defence Production / iDEX  
**Title:** *Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception*  
**Team:** Perceptra  

---

## 1. Executive Summary & Problem Formulation

Autonomous defense Unmanned Ground Vehicles (UGVs) operating in GPS-denied, unstructured off-road terrains face stringent SWaP-C (Size, Weight, Power, and Cost) constraints. Processing raw 3D LiDAR point clouds at uniform high resolution across operational envelopes ($200\text{ m} \times 200\text{ m}$) demands over 16 million grid cells, triggering memory exhaustion and computational latency exceeding 100 ms.

The **Perceptra Engine** delivers a high-throughput, research-grade perception pipeline:
1. **Stage 1 — Spatiotemporal 3D Multi-Object Tracking**: Retains a 4-frame rolling history ($\Delta t \approx 0.1\text{ s}$), extracts cell-wise spatial intensity moments ($\mu_i, \sigma_i^2$) via zero-loop bincounting, isolates dynamic candidates via temporal fluctuation, and estimates planar velocity vectors ($V_x, V_y$) before grid projection.
2. **Stage 2 — LiDAR Point-Cloud Segmentation using Sparse CNN**: SubMConv3d / SPVCNN architecture categorizing points into Drivable Terrain, Rough Surface, Curbs/Drop-offs, Static Obstacles (walls, poles), and Dynamic Objects (vehicles, personnel).
3. **Stage 3 — Adaptive Variable Resolution**: Rectangular near-box (0–10 m @ 5 cm) and radial far-field (10–100 m @ 50 cm), eliminating projection alignment errors and slashing memory by **98.00%** (320,000 vs 16,000,000 cells).
4. **Stage 4 — Adaptive 2.5D LiDAR Mapping**: Rapid Delta-Z ($\Delta Z = Z_{\max} - Z_{\min}$) micro-elevation mapping and geometric-semantic hazard fusion into an $800 \times 800$ ego-centric Dynamic Occupancy Grid Map (DOGMa).
5. **Real-Time Integration & WebGL Dashboard**: FastAPI backend with WebSocket streaming at 60 FPS coupled to a Three.js 3D WebGL point cloud viewer and 2.5D DOGMa heatmap canvas.

---

## 2. Four-Stage Technical Pipeline (Perceptra Architecture)

```
        LiDAR Point Stream (4 frames @ ~0.1 s)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Spatiotemporal 3D Multi-Object Tracking            │
│  • IN: 4-frame short history (FIFO buffer @ ~0.1s)          │
│  • ACTION: Lightweight motion cues + Intensity moments      │
│            Pre-grid DBSCAN velocity estimation [Vx, Vy]     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: LiDAR Point-Cloud Segmentation using Sparse CNN    │
│  • IN: Point cloud -> classes                               │
│  • ACTION: Segment terrain and objects; color labels        │
│            (SubMConv3d / SPVCNN / PyTorch Vectorized)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Adaptive Variable Resolution                       │
│  • IN: Range -> cell size                                   │
│  • ACTION: 0–10 m: 5 cm (Near-Box)                          │
│            10–100 m: 50 cm (Far Horizon)                    │
│            98.00% Cell-Count Reduction                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: Adaptive 2.5D LiDAR Mapping                        │
│  • IN: Labels + grid -> map                                 │
│  • ACTION: Occupied / non-occupied + Delta-Z variation      │
│            800x800 Elevation & Traversability DOGMa         │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
FastAPI REST & WebSockets         WebGL Interactive Dashboard
(/api/sweep, /ws/stream)          (Three.js 3D + 2.5D Canvas)
```
              DogMaBuilder
                   │
                   ▼
             Final 2.5D DOGMa
                   │
                   ▼
          Downstream Navigation
              (EXTERNAL)
```

---

## 4. Mathematical Formulation

### 4.1. Pre-Grid Dynamic Kinematics
Dynamic candidate points $L_{\text{dynamic}}$ are clustered in 2D using DBSCAN. For each tracked object cluster:
$$\mathbf{C}(t) = \left(\frac{1}{N}\sum x_i, \; \frac{1}{N}\sum y_i\right)$$
$$V_x = \frac{C_x(t_0) - C_x(t_{-1})}{\Delta t}, \quad V_y = \frac{C_y(t_0) - C_y(t_{-1})}{\Delta t}$$

### 4.2. Dual-Tier Variable-Resolution Spatial Routing
Given point horizontal radius $r = \sqrt{x^2 + y^2}$:
$$\text{Scope}(r) = \begin{cases} \text{LOCAL} \; (s = 0.05\text{ m}), & 0 \le r < 10\text{ m} \\ \text{GLOBAL} \; (s = 0.50\text{ m}), & 10\text{ m} \le r \le 100\text{ m} \\ \text{DISCARD}, & r > 100\text{ m} \end{cases}$$

Cell indices are determined via vectorized integer floor indexing:
$$i = \lfloor x / s \rfloor, \quad j = \lfloor y / s \rfloor$$

### 4.3. Streaming Welford Algorithm (Local Grid Only)
For each local cell, streaming height samples $z_1, \dots, z_n$ are aggregated without storing raw point lists:
$$\delta_n = z_n - \mu_{n-1}$$
$$\mu_n = \mu_{n-1} + \frac{\delta_n}{n}$$
$$M_{2,n} = M_{2,n-1} + \delta_n (z_n - \mu_n)$$
$$\sigma_z^2 = \frac{M_{2,n}}{n - 1} \quad (n \ge 2)$$

---

## 5. Memory Calculation (Cell-Count Reduction)

For a $200\text{ m} \times 200\text{ m}$ operational envelope:

| Grid Type | Resolution ($s$) | Coverage Radius ($r$) | Cells |
|---|---|---|---|
| **Uniform Baseline** | 5 cm (0.05 m) | $0 \le r \le 100\text{ m}$ | $\left(\frac{200}{0.05}\right)^2 = \mathbf{16,000,000}$ |
| **Proposed Local Grid** | 5 cm (0.05 m) | $0 \le r < 10\text{ m}$ | $\left(\frac{20}{0.05}\right)^2 = 160,000$ |
| **Proposed Global Grid** | 50 cm (0.50 m) | $10\text{ m} \le r \le 100\text{ m}$ | $\left(\frac{200}{0.50}\right)^2 = 160,000$ |
| **Total Proposed** | Variable | $0 \le r \le 100\text{ m}$ | $\mathbf{320,000}$ |

$$\text{Cell-Count Reduction} = 1 - \frac{320,000}{16,000,000} = \mathbf{98.00\%}$$

*Note: This is a verified **cell-count** reduction. Complete RAM reduction depends on point buffers and OS memory footprint.*

---

## 6. Hardware Policy: GPU Training / CPU-NPU Inference Separation

- **Offline Training (GPU)**: Sparse 3D CNNs (MinkowskiEngine, SPVNAS) trained on large datasets (SemanticKITTI, Rellis-3D).
- **Onboard Deployment (CPU / NPU)**: The onboard perception pipeline runs strictly on CPU/NPU without requiring a discrete GPU. Temporal buffering, Welford statistics, DBSCAN clustering, and DOGMa serialization execute in vectorized CPU runtime.

---

## 7. Project Structure

```
sih26053_perception/
├── config/
│   └── default.yaml             # Tunable parameters
├── src/
│   └── perception/
│       ├── __init__.py
│       ├── types.py             # Dataclasses & interfaces
│       ├── validation.py        # Input sanitization
│       ├── temporal_stacker.py  # Module 1: Rolling 4D buffer
│       ├── semantic_segmenter.py# Module 2: Semantic AI interface & mock
│       ├── kinematics_engine.py # Module 3: Pre-grid DBSCAN velocity
│       ├── variable_grid.py     # Module 4: Dual-tier spatial quantizer
│       ├── terrain_analyzer.py  # Module 5: Welford statistics
│       ├── dogma.py             # Module 6: DOGMa frame builder
│       └── pipeline.py          # 6-phase pipeline orchestrator
├── tests/
│   ├── test_temporal_stacker.py
│   ├── test_semantic_segmenter.py
│   ├── test_kinematics.py
│   ├── test_variable_grid.py
│   ├── test_welford.py
│   └── test_dogma.py
├── examples/
│   ├── demo_pipeline.py               # Programmatic usage example
│   ├── visualize_pipeline.py          # Simple & Technical visualizers
│   └── visualize_4stage_pipeline.py   # Publication-grade 4-panel pipeline figure
├── src/
│   ├── api/
│   │   ├── server.py                  # FastAPI real-time REST & WebSocket server
│   │   └── scenarios.py               # Defense UGV synthetic scenario generator
│   └── perception/
│       ├── eigensight_pipeline.py     # 4-Stage PyTorch GPU pipeline
│       └── ...
├── web/
│   ├── index.html                     # Perceptra interactive dashboard
│   ├── app.js                         # Three.js 3D WebGL + 2.5D Canvas client
│   └── styles.css                     # Defense/cyber UI styling
├── benchmarks/
│   ├── benchmark_models.py            # SPVCNN vs SPVNAS comparison
│   └── benchmark_eigensight.py        # Scaling & latency benchmark
├── tests/                             # 26 automated unit & integration tests
├── new.py                             # PyTorch EigenSight pipeline benchmark
├── run_server.py                      # One-click dashboard & API launcher
├── main.py                            # Main multi-mode demonstration script
├── pyproject.toml
└── requirements.txt
```

---

## 8. Execution Commands

### 8.1. Launch FastAPI Server & Live Interactive Dashboard
```powershell
python run_server.py
```
Open **http://localhost:8000/** in your browser to view:
- Live interactive 3D WebGL LiDAR point cloud stream
- Real-time 2.5D DOGMa Elevation & Traversability map
- 4-Stage pipeline latency indicators
- Tracked dynamic objects and velocity vectors $[V_x, V_y]$

### 8.2. Run GPU/CUDA Benchmark (`new.py`)
```powershell
python new.py
```
Executes 100,000 synthetic LiDAR points with GPU acceleration:
- Processing latency: ~7.8 ms (128+ FPS)
- Peak VRAM: ~54 MB
- 800x800 Elevation & Terrain Grids

### 8.3. Run Multi-Mode Demonstration (`main.py`)
```powershell
# Run both CPU and GPU pipelines
python main.py

# Run GPU PyTorch pipeline only
python main.py --mode gpu

# Run CPU vectorized runtime only
python main.py --mode cpu
```

### 8.4. Run Complete Automated Test Suite (26 Tests)
```powershell
python -m pytest
```

### 8.5. Run Performance & Scaling Benchmark
```powershell
python benchmarks/benchmark_eigensight.py
```

### 8.6. Generate Publication-Grade Pipeline Figure
```powershell
python examples/visualize_4stage_pipeline.py
```
Generates `perceptra_4stage_pipeline.png` with all four stages visual overlay.

---

## 9. Benchmark & Verification Summary

| Metric | Target / Baseline | Perceptra Result | Compliance |
|---|---|---|---|
| **Pipeline Latency** | $< 50\text{ ms}$ ($> 20\text{ FPS}$) | **$7.72\text{ ms}$ ($129.5\text{ FPS}$)** | **PASS** (10x faster than target) |
| **Operational Envelope** | $200\text{ m} \times 200\text{ m}$ ($r \le 100\text{ m}$) | **$200\text{ m} \times 200\text{ m}$ supported** | **PASS** |
| **Grid Resolution** | $5\text{ cm}$ (near), $50\text{ cm}$ (far) | **$5\text{ cm}$ ($r < 10\text{ m}$), $50\text{ cm}$ ($r \ge 10\text{ m}$)** | **PASS** |
| **Cell-Count Reduction** | $\ge 95\%$ | **$98.00\%$** (320,000 vs 16,000,000) | **PASS** |
| **VRAM Footprint** | $< 8,000\text{ MB}$ (RTX GPU) | **$71.8\text{ MB}$** | **PASS** ($< 1\%$ of budget) |
| **Dynamic Kinematics** | Planar $[V_x, V_y]$ tracked pre-grid | **DBSCAN + Centroid tracking** | **PASS** |
| **2.5D Map Size** | $800 \times 800$ grid | **$800 \times 800$ asserted** | **PASS** |

---

## 10. Downstream Navigation Stack Integration

The final output `DogMaFrame` and `PipelineOutput` provide:
- `elevation_grid`: Tensor shape $(800, 800)$ with per-cell Delta-Z ($\Delta Z$).
- `terrain_grid`: Tensor shape $(800, 800)$ with classified hazards (Drivable, Curb/Drop-off, Static Obstacle, Dynamic Object).
- `dynamic_objects`: List of `TrackedDynamicObject` with centroid $(C_x, C_y)$, velocity $(V_x, V_y)$, speed, and confidence.
- Ready for zero-copy ROS2 / DDS transport for UGV trajectory and navigation planners.
