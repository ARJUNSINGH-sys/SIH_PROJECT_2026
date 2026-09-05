/**
 * EIGENSIGHT — Client Perception Engine & Navigation Cockpit
 * Problem Statement #26053 (DRDO / iDEX)
 * Features:
 *   1. Authentic Eigensight Hero Foveated Grid Simulation (from eigensight.netlify.app)
 *   2. 3D WebGL Ego-Centric Point Cloud Stream (Three.js)
 *   3. De-Cluttered Dual-Tier 2.5D DOGMa & Foveated Grid Viewer (Zero text clutter!)
 *   4. Rover Attitude & Heading Telemetry (Yaw Compass, Pitch/Roll Inclinometer)
 *   5. Interactive 7,000-Picture 7-Fold CV Training Metrics Modal
 */

// Safe roundRect polyfill for all browsers and WebViews
if (typeof CanvasRenderingContext2D !== "undefined" && !CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    r = typeof r === "number" ? r : (Array.isArray(r) ? r[0] : 0) || 0;
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    this.beginPath();
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    this.closePath();
    return this;
  };
}

// Application State
const state = {
  isPlaying: true,
  isAutoPatrol: true,
  currentScenario: "eigensight_stacking",
  currentFrame: 0,
  colorMode: "class",
  dogmaLayer: "std",
  lastFrameData: null,
  rover: {
    x: 0.0,
    y: 0.0,
    z: 0.0,
    yaw_deg: 45.0,
    pitch_deg: 2.5,
    roll_deg: -1.2,
    speed_mps: 2.4,
    steering_deg: 0.0,
  },
  hoveredDynamicObject: null,
};

// =====================================================================
// 1. HERO CANVAS ANIMATION (From eigensight.netlify.app)
// =====================================================================
function initHeroCanvas() {
  const canvas = document.getElementById("grid-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let w, h, dpr;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth;
    h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);
  resize();

  const cx = () => w / 2;
  const cy = () => h / 2;
  const colors = { terrain: "#E8A34D", static: "#6E85A0", dynamic: "#47D7E3", line: "#1C232E" };

  const rand = (a, b) => a + Math.random() * (b - a);
  const points = [];
  for (let i = 0; i < 340; i++) {
    const x = rand(-1, 1);
    const y = rand(-1, 1);
    const d = Math.max(Math.abs(x), Math.abs(y));
    points.push({ x, y, d });
  }

  const obstacles = [
    { x: -0.55, y: -0.15, size: 5 },
    { x: -0.62, y: 0.18, size: 4 },
    { x: 0.5, y: -0.42, size: 6 },
    { x: 0.25, y: 0.45, size: 5 },
  ];
  let dynT = 0;

  function squarePath(t, s) {
    t = t % 4;
    if (t < 1) return { x: -s + 2 * s * t, y: -s };
    if (t < 2) return { x: s, y: -s + 2 * s * (t - 1) };
    if (t < 3) return { x: s - 2 * s * (t - 2), y: s };
    return { x: -s, y: s - 2 * s * (t - 3) };
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    const maxR = Math.min(w, h) / 2 - 14;

    // Coarse global square grid
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    const coarseDivs = 6;
    for (let i = 0; i <= coarseDivs; i++) {
      const t = -1 + (2 * i) / coarseDivs;
      ctx.beginPath();
      ctx.moveTo(cx() + t * maxR, cy() - maxR);
      ctx.lineTo(cx() + t * maxR, cy() + maxR);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx() - maxR, cy() + t * maxR);
      ctx.lineTo(cx() + maxR, cy() + t * maxR);
      ctx.stroke();
    }

    // Nested square resolution zone boundaries
    ctx.strokeStyle = "rgba(71,215,227,0.35)";
    ctx.lineWidth = 1.2;
    [0.28, 0.55].forEach(t => {
      ctx.strokeRect(cx() - maxR * t, cy() - maxR * t, maxR * t * 2, maxR * t * 2);
    });

    // Fine local square grid inside innermost zone (foveated patch)
    const fineT = 0.28;
    const fineDivs = 9;
    ctx.strokeStyle = "rgba(71,215,227,0.18)";
    ctx.lineWidth = 0.6;
    for (let i = 0; i <= fineDivs; i++) {
      const t = -fineT + (2 * fineT * i) / fineDivs;
      ctx.beginPath();
      ctx.moveTo(cx() + t * maxR, cy() - fineT * maxR);
      ctx.lineTo(cx() + t * maxR, cy() + fineT * maxR);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx() - fineT * maxR, cy() + t * maxR);
      ctx.lineTo(cx() + fineT * maxR, cy() + t * maxR);
      ctx.stroke();
    }

    // Terrain points (coarser / larger toward edges)
    points.forEach(p => {
      const x = cx() + p.x * maxR;
      const y = cy() + p.y * maxR;
      const size = 0.9 + p.d * 2.4;
      ctx.fillStyle = `rgba(232,163,77,${0.15 + (1 - p.d) * 0.35})`;
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
    });

    // Static obstacles
    obstacles.forEach(o => {
      const x = cx() + o.x * maxR;
      const y = cy() + o.y * maxR;
      ctx.fillStyle = colors.static;
      ctx.fillRect(x - o.size / 2, y - o.size / 2, o.size, o.size * 1.6);
    });

    // Dynamic object patrolling square path with glow
    const ds = 0.5;
    const dp = squarePath(dynT, ds);
    const dx = cx() + dp.x * maxR;
    const dy = cy() + dp.y * maxR;
    ctx.fillStyle = colors.dynamic;
    ctx.shadowColor = colors.dynamic;
    ctx.shadowBlur = 8;
    ctx.fillRect(dx - 4, dy - 4, 8, 8);
    ctx.shadowBlur = 0;

    // Faint motion trail along the square path
    ctx.strokeStyle = "rgba(71,215,227,0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const trailStart = squarePath(dynT - 0.5, ds);
    ctx.moveTo(cx() + trailStart.x * maxR, cy() + trailStart.y * maxR);
    for (let s = 0.1; s <= 0.5; s += 0.1) {
      const pt = squarePath(dynT - 0.5 + s, ds);
      ctx.lineTo(cx() + pt.x * maxR, cy() + pt.y * maxR);
    }
    ctx.stroke();

    // Center sensor marker
    ctx.fillStyle = "#E7ECF2";
    ctx.fillRect(cx() - 2.5, cy() - 2.5, 5, 5);

    if (!reduceMotion) {
      dynT += 0.0026;
      requestAnimationFrame(draw);
    }
  }
  draw();
}

// =====================================================================
// 2. THREE.JS 3D ROVER & POINT CLOUD RENDERER
// =====================================================================
let scene, camera, renderer, controls;
let pointCloudMesh = null;
let rover3DGroup = null;

function initThreeJS() {
  const container = document.getElementById("threejs-container");
  if (!container) return;

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 480;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080B0F);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 500);
  camera.position.set(-16, -20, 14);
  camera.up.set(0, 0, 1);

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  if (window.THREE && THREE.OrbitControls) {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);
  }

  // Ground Grid
  const gridHelper = new THREE.GridHelper(50, 50, 0x47D7E3, 0x1C232E);
  gridHelper.rotation.x = Math.PI / 2;
  scene.add(gridHelper);

  // 3D Rover Chassis
  rover3DGroup = new THREE.Group();

  const bodyGeo = new THREE.BoxGeometry(2.4, 1.4, 0.6);
  const bodyMat = new THREE.MeshBasicMaterial({ color: 0x6E85A0 });
  const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
  bodyMesh.position.z = 0.5;
  rover3DGroup.add(bodyMesh);

  const cabinGeo = new THREE.BoxGeometry(1.2, 1.1, 0.5);
  const cabinMat = new THREE.MeshBasicMaterial({ color: 0x47D7E3 });
  const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
  cabinMesh.position.set(-0.2, 0, 0.9);
  rover3DGroup.add(cabinMesh);

  const wheelGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.3, 16);
  const wheelMat = new THREE.MeshBasicMaterial({ color: 0x0A0D12 });
  [[-0.8, -0.8], [-0.8, 0.8], [0.8, -0.8], [0.8, 0.8]].forEach(([wx, wy]) => {
    const wheel = new THREE.Mesh(wheelGeo, wheelMat);
    wheel.rotation.z = Math.PI / 2;
    wheel.position.set(wx, wy, 0.35);
    rover3DGroup.add(wheel);
  });

  scene.add(rover3DGroup);

  // 1. Local 10m Cubical Bounding Box ([-10m, +10m]^2 x [-0.5m, +2.5m]) - IEEE Paper Architecture
  const localCubeGeo = new THREE.BoxGeometry(20.0, 20.0, 3.0);
  const localCubeEdges = new THREE.EdgesGeometry(localCubeGeo);
  const localCubeMat = new THREE.LineBasicMaterial({
    color: 0x47D7E3,
    transparent: true,
    opacity: 0.65,
    linewidth: 1.5,
  });
  const localCubeMesh = new THREE.LineSegments(localCubeEdges, localCubeMat);
  localCubeMesh.position.set(0, 0, 1.0);
  scene.add(localCubeMesh);

  // 2. Tier-1 20m Partition Cubical Boundary ([-20m, +20m]^2 x [-1m, +4m]) - Buerkle et al. (2020)
  const tier1CubeGeo = new THREE.BoxGeometry(40.0, 40.0, 5.0);
  const tier1CubeEdges = new THREE.EdgesGeometry(tier1CubeGeo);
  const tier1CubeMat = new THREE.LineBasicMaterial({
    color: 0xA855F7,
    transparent: true,
    opacity: 0.35,
  });
  const tier1CubeMesh = new THREE.LineSegments(tier1CubeEdges, tier1CubeMat);
  tier1CubeMesh.position.set(0, 0, 1.5);
  scene.add(tier1CubeMesh);

  window.addEventListener("resize", () => {
    if (!container) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

function updateThreePointCloud(pointsSample, colorMode) {
  if (!scene || !pointsSample || pointsSample.length === 0) return;

  const n = pointsSample.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  // High-contrast class colors:
  // Class 0: Terrain / Dirt Path (#E8A34D) -> rgb(0.91, 0.64, 0.30)
  // Class 1: Rough / Curb Hazard (#F59E0B) -> rgb(0.96, 0.62, 0.04)
  // Class 2: Static Obstacles (#EF4444) -> rgb(0.94, 0.27, 0.27)
  // Class 3: Dynamic Object (#47D7E3) -> rgb(0.28, 0.84, 0.89)
  const classColors = [
    [0.91, 0.64, 0.30],
    [0.96, 0.62, 0.04],
    [0.94, 0.27, 0.27],
    [0.28, 0.84, 0.89],
  ];

  for (let i = 0; i < n; i++) {
    const pt = pointsSample[i];
    positions[i * 3] = pt[0];
    positions[i * 3 + 1] = pt[1];
    positions[i * 3 + 2] = pt[2];

    if (colorMode === "elevation") {
      // Vibrant Spectral/Turbo palette: Blue -> Cyan -> Green -> Yellow -> Red
      const t = Math.max(0, Math.min(1, (pt[2] + 0.15) / 1.6));
      if (t < 0.25) {
        colors[i * 3] = 0.1;
        colors[i * 3 + 1] = 0.4 + t * 2.4;
        colors[i * 3 + 2] = 1.0;
      } else if (t < 0.5) {
        colors[i * 3] = 0.1 + (t - 0.25) * 3.6;
        colors[i * 3 + 1] = 1.0;
        colors[i * 3 + 2] = 1.0 - (t - 0.25) * 4.0;
      } else if (t < 0.75) {
        colors[i * 3] = 1.0;
        colors[i * 3 + 1] = 1.0 - (t - 0.5) * 2.0;
        colors[i * 3 + 2] = 0.1;
      } else {
        colors[i * 3] = 1.0;
        colors[i * 3 + 1] = 0.5 - (t - 0.75) * 2.0;
        colors[i * 3 + 2] = 0.2;
      }
    } else if (colorMode === "intensity") {
      const normI = Math.max(0.15, Math.min(1.0, (pt[3] || 50) / 220));
      colors[i * 3] = normI;
      colors[i * 3 + 1] = normI * 0.9;
      colors[i * 3 + 2] = normI * 0.7;
    } else {
      let cls = 0;
      if (pt[2] > 0.35) cls = 2; // Static
      else if (pt[2] > 0.06) cls = 1; // Rough / curb
      const c = classColors[cls];
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
    }
  }

  if (pointCloudMesh) {
    scene.remove(pointCloudMesh);
    pointCloudMesh.geometry.dispose();
    pointCloudMesh.material.dispose();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  // Solid, crisp point rendering
  const material = new THREE.PointsMaterial({
    size: 2.8,
    vertexColors: true,
    sizeAttenuation: false, // Ensures points remain crisp and solid at any distance
    transparent: true,
    opacity: 0.92,
  });

  pointCloudMesh = new THREE.Points(geometry, material);
  scene.add(pointCloudMesh);

  // Update Rover 3D Pose
  if (rover3DGroup && state.rover) {
    rover3DGroup.position.set(0, 0, 0);
    const yawRad = ((-state.rover.yaw_deg + 90) * Math.PI) / 180.0;
    rover3DGroup.rotation.z = yawRad;
  }
}

// =====================================================================
// 3. DE-CLUTTERED DUAL-TIER 2.5D DOGMA & FOVEATED MAP
// =====================================================================
function initDogmaCanvas() {
  const canvas = document.getElementById("dogma-canvas");
  if (!canvas) return;

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const ppm = canvas.width / 40.0;

    if (state.lastFrameData && state.lastFrameData.dynamic_objects) {
      const hovered = state.lastFrameData.dynamic_objects.find(obj => {
        const dpx = cx + (obj.x - state.rover.x) * ppm;
        const dpy = cy - (obj.y - state.rover.y) * ppm;
        return Math.hypot(mx - dpx, my - dpy) < 14;
      });
      state.hoveredDynamicObject = hovered || null;
    }
  });

  canvas.addEventListener("mouseleave", () => {
    state.hoveredDynamicObject = null;
  });
}

function drawDogma2D(data) {
  const canvas = document.getElementById("dogma-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const w = canvas.parentElement.clientWidth || 600;
  const h = canvas.parentElement.clientHeight || 480;
  canvas.width = w;
  canvas.height = h;

  ctx.fillStyle = "#0A0D12";
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;
  const mapSpan = 40.0; // [-20m, +20m]
  const ppm = w / mapSpan;

  const rover = data.rover || state.rover;

  // 1. Draw Far-Field Occupancy & Dempster-Shafer Objects
  // Only render high-confidence dynamic (Blue) or static (Red) cells in the far-field
  if (data.belief_map && data.belief_map.dynamic && data.belief_map.static) {
    const dynGrid = data.belief_map.dynamic;
    const staGrid = data.belief_map.static;
    const rows = dynGrid.length;
    const cols = dynGrid[0].length;
    const cellW = w / cols;
    const cellH = h / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const mD = dynGrid[r][c];
        const mS = staGrid[r][c];
        const px = c * cellW;
        const py = r * cellH;
        const dist = Math.hypot(px - cx, py - cy) / ppm;

        if (dist > 10.0) { // Only outside 10m fovea
          if (mD > mS && mD > 0.18) {
            ctx.fillStyle = `rgba(59, 130, 246, ${Math.min(0.9, mD * 1.4)})`;
            ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
          } else if (mS > 0.20) {
            ctx.fillStyle = `rgba(239, 68, 68, ${Math.min(0.85, mS * 1.3)})`;
            ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
          }
        }
      }
    }
  } else if (data.global_binary_occupancy) {
    const occGrid = data.global_binary_occupancy;
    const rows = occGrid.length;
    const cols = occGrid[0].length;
    const cellW = w / cols;
    const cellH = h / rows;

    ctx.fillStyle = "rgba(110, 133, 160, 0.35)";
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (occGrid[r][c] === 1) {
          const px = c * cellW;
          const py = r * cellH;
          const dist = Math.hypot(px - cx, py - cy) / ppm;
          if (dist > 10.0) {
            ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
          }
        }
      }
    }
  }

  // 2. Draw 9-Partition Cubical Grid Wireframe (Buerkle et al. IEEE IV 2020 Fig. 2)
  const t1Span = 40.0 * ppm; // 40m x 40m center box [-20m, +20m]
  const t1X0 = cx - t1Span / 2;
  const t1Y0 = cy - t1Span / 2;

  // Grid wireframe lines for partitions
  ctx.strokeStyle = "rgba(71, 215, 227, 0.08)";
  ctx.lineWidth = 1;
  const fineStep = 2.0 * ppm; // 2m grid markers
  for (let gx = 0; gx <= w; gx += fineStep) {
    ctx.beginPath();
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, h);
    ctx.stroke();
  }
  for (let gy = 0; gy <= h; gy += fineStep) {
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(w, gy);
    ctx.stroke();
  }

  // 3. Draw Local 10m Cubical Foveated Micro-Terrain Grid (5cm resolution)
  let targetGrid = data.local_std_heatmap;
  if (state.dogmaLayer === "variance") targetGrid = data.local_var_heatmap;

  const fovSpan = 20.0 * ppm; // 20m x 20m box [-10m, +10m]
  const fovX0 = cx - fovSpan / 2;
  const fovY0 = cy - fovSpan / 2;

  if (targetGrid && targetGrid.length > 0) {
    const lRows = targetGrid.length;
    const lCols = targetGrid[0].length;
    const lCellW = fovSpan / lCols;
    const lCellH = fovSpan / lRows;

    ctx.save();
    ctx.beginPath();
    ctx.rect(fovX0, fovY0, fovSpan, fovSpan);
    ctx.clip();

    for (let r = 0; r < lRows; r++) {
      for (let c = 0; c < lCols; c++) {
        const val = targetGrid[r][c];
        if (val > 0.001) {
          if (val >= 0.08) {
            ctx.fillStyle = "rgba(245, 158, 11, 0.90)"; // High Roughness / Curb Hazard
          } else if (val >= 0.04) {
            ctx.fillStyle = "rgba(232, 163, 77, 0.70)"; // Mild Undulation
          } else {
            ctx.fillStyle = "rgba(52, 211, 153, 0.50)";  // Smooth Drivable Ground
          }
          ctx.fillRect(fovX0 + c * lCellW, fovY0 + r * lCellH, lCellW + 0.5, lCellH + 0.5);
        }
      }
    }
    ctx.restore();
  }

  // 4. Draw Cubical Bounding Frames
  // Local 10m Cubical Fovea Frame (Cyan Dashed Box)
  ctx.strokeStyle = "rgba(71, 215, 227, 0.9)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(fovX0, fovY0, fovSpan, fovSpan);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(71, 215, 227, 0.95)";
  ctx.font = "600 10px 'IBM Plex Mono', monospace";
  ctx.fillText("10m LOCAL CUBICAL FOVEA (5cm Cells)", fovX0 + 6, fovY0 + 13);

  // Tier-1 20m Partition Boundary Frame (Purple Dashed Box)
  ctx.strokeStyle = "rgba(168, 85, 247, 0.75)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(t1X0, t1Y0, t1Span, t1Span);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(168, 85, 247, 0.95)";
  ctx.font = "600 10px 'IBM Plex Mono', monospace";
  ctx.fillText("TIER-1 PARTITION X4 (0.1m Cells - Buerkle et al.)", t1X0 + 8, t1Y0 + 13);

  // Partition Labels X0..X8 from Paper Fig. 2
  ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
  ctx.font = "500 9px 'IBM Plex Mono', monospace";
  ctx.fillText("X1: MID (0.2m)", cx - 35, Math.max(12, t1Y0 - 6));
  ctx.fillText("X7: MID (0.2m)", cx - 35, Math.min(h - 6, t1Y0 + t1Span + 14));
  ctx.fillText("X3: MID (0.2m)", 6, cy);
  ctx.fillText("X5: MID (0.2m)", w - 75, cy);
  ctx.fillText("X0: COARSE (0.4m)", 6, 14);
  ctx.fillText("X2: COARSE (0.4m)", w - 95, 14);
  ctx.fillText("X6: COARSE (0.4m)", 6, h - 8);
  ctx.fillText("X8: COARSE (0.4m)", w - 95, h - 8);

  // Animated Scan Sweep Line
  const sweepScanY = (Date.now() % 3200) / 3200 * h;
  const sweepGrad = ctx.createLinearGradient(0, sweepScanY - 14, 0, sweepScanY);
  sweepGrad.addColorStop(0, "rgba(71, 215, 227, 0)");
  sweepGrad.addColorStop(1, "rgba(71, 215, 227, 0.14)");
  ctx.fillStyle = sweepGrad;
  ctx.fillRect(0, sweepScanY - 14, w, 14);
  ctx.strokeStyle = "rgba(71, 215, 227, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, sweepScanY);
  ctx.lineTo(w, sweepScanY);
  ctx.stroke();

  // 5. Draw Eigensight Stacking Demo Elements (Dirt Path, Tree, Wall, Cars)
  try {
    if (data.meta && data.meta.title === "TEMPORAL STACKING") {
      // A. Curved Dirt Path
      ctx.strokeStyle = "rgba(146, 64, 14, 0.55)";
      ctx.lineWidth = 3.2 * ppm;
      ctx.beginPath();
      for (let y = -14; y <= 14; y += 0.5) {
        const px = cx + (0.8 * Math.sin(y * 0.22) + 0.05 * y - rover.x) * ppm;
        const py = cy - (y - rover.y) * ppm;
        if (y === -14) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // B. Green Tree Canopy at bottom-left
      if (data.meta.tree_pos) {
        const tx = cx + (data.meta.tree_pos.x - rover.x) * ppm;
        const ty = cy - (data.meta.tree_pos.y - rover.y) * ppm;
        ctx.fillStyle = "rgba(34, 197, 94, 0.85)";
        ctx.strokeStyle = "#15803D";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(tx, ty, 3.2 * ppm, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#DCFCE7";
        ctx.font = "bold 9px 'IBM Plex Mono', monospace";
        ctx.fillText("tree (STATIC)", tx - 28, ty + 3);
      }

      // C. Stone Lodge Wall at top
      if (data.meta.lodge_wall_y !== undefined) {
        const wy = cy - (data.meta.lodge_wall_y - rover.y) * ppm;
        ctx.fillStyle = "rgba(100, 116, 139, 0.85)";
        ctx.strokeStyle = "#CBD5E1";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(cx - 7 * ppm, wy - 1.2 * ppm, 14 * ppm, 2.4 * ppm, 4);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#F8FAFC";
        ctx.font = "bold 9px 'IBM Plex Mono', monospace";
        ctx.fillText("lodge wall (STATIC)", cx - 45, wy + 3);
      }

      // D. Parked RED CAR (Static Vehicle)
      if (data.meta.red_car_pos) {
        const rx = cx + (data.meta.red_car_pos.x - rover.x) * ppm;
        const ry = cy - (data.meta.red_car_pos.y - rover.y) * ppm;
        ctx.fillStyle = "#EF4444";
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(rx - 0.9 * ppm, ry - 1.8 * ppm, 1.8 * ppm, 3.6 * ppm, 3);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#FEE2E2";
        ctx.font = "bold 9px 'IBM Plex Mono', monospace";
        ctx.fillText("RED CAR (STATIC)", rx + 1.2 * ppm, ry);
      }

      // E. Yellow Hazard Box
      const ybx = cx + (0.1 - rover.x) * ppm;
      const yby = cy - (-0.8 - rover.y) * ppm;
      ctx.fillStyle = "#FBBF24";
      ctx.strokeStyle = "#B45309";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(ybx - 0.7 * ppm, yby - 0.7 * ppm, 1.4 * ppm, 1.4 * ppm, 2);
      ctx.fill();
      ctx.stroke();

      // F. CYAN CAR (Translating Dynamic Vehicle)
      if (data.meta.cyan_car_pos) {
        const cyx = cx + (data.meta.cyan_car_pos.x - rover.x) * ppm;
        const cyy = cy - (data.meta.cyan_car_pos.y - rover.y) * ppm;
        ctx.fillStyle = "#47D7E3";
        ctx.shadowColor = "#47D7E3";
        ctx.shadowBlur = 12;
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(cyx - 0.9 * ppm, cyy - 1.7 * ppm, 1.8 * ppm, 3.4 * ppm, 3);
        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Velocity arrow
        ctx.strokeStyle = "#47D7E3";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cyx, cyy - 1.7 * ppm);
        ctx.lineTo(cyx, cyy - 3.8 * ppm);
        ctx.stroke();

        ctx.fillStyle = "#47D7E3";
        ctx.font = "bold 9.5px 'IBM Plex Mono', monospace";
        ctx.fillText("CYAN CAR (DYNAMIC)", cyx + 1.2 * ppm, cyy);
      }
    }
  } catch (err) {
    console.warn("Stacking demo elements error:", err);
  }

  // 6. Draw Tracked Dynamic Object Nodes
  if (data.dynamic_objects) {
    data.dynamic_objects.forEach((obj, idx) => {
      const dpx = cx + (obj.x - rover.x) * ppm;
      const dpy = cy - (obj.y - rover.y) * ppm;

      ctx.fillStyle = "#47D7E3";
      ctx.shadowColor = "#47D7E3";
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(dpx, dpy, 5.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#FFFFFF";
      ctx.beginPath();
      ctx.arc(dpx, dpy, 2.5, 0, 2 * Math.PI);
      ctx.fill();

      const speed = Math.max(10, obj.speed * 3.5);
      const angle = Math.atan2(-obj.vy, obj.vx);
      const tox = dpx + Math.cos(angle) * speed;
      const toy = dpy + Math.sin(angle) * speed;

      ctx.strokeStyle = "rgba(71, 215, 227, 0.8)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(dpx, dpy);
      ctx.lineTo(tox, toy);
      ctx.stroke();
    });
  }

  // 7. Draw Rover Chassis at Center (0, 0)
  drawEgoRover2D(ctx, cx, cy, rover.yaw_deg, rover.steering_deg);
}

function drawEgoRover2D(ctx, cx, cy, yawDeg, steerDeg) {
  ctx.save();
  ctx.translate(cx, cy);
  const yawRad = (yawDeg * Math.PI) / 180.0;
  ctx.rotate(yawRad);

  // Headlight illumination beam
  ctx.fillStyle = "rgba(232, 163, 77, 0.12)";
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-20, -75);
  ctx.lineTo(20, -75);
  ctx.closePath();
  ctx.fill();

  // Rover Body Chassis (Slate gray)
  ctx.fillStyle = "#6E85A0";
  ctx.strokeStyle = "#1C232E";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(-10, -16, 20, 32, 4);
  ctx.fill();
  ctx.stroke();

  // Cabin / Sensor Dome (Cyan)
  ctx.fillStyle = "#47D7E3";
  ctx.beginPath();
  ctx.arc(0, -2, 5.5, 0, 2 * Math.PI);
  ctx.fill();

  // Direction Pointer
  ctx.fillStyle = "#FFFFFF";
  ctx.beginPath();
  ctx.moveTo(0, -18);
  ctx.lineTo(-4, -12);
  ctx.lineTo(4, -12);
  ctx.closePath();
  ctx.fill();

  // Wheels
  ctx.fillStyle = "#0A0D12";
  [[-12, -12], [10, -12], [-12, 8], [10, 8]].forEach(([wx, wy], i) => {
    ctx.save();
    ctx.translate(wx, wy);
    if (i < 2) ctx.rotate((steerDeg * Math.PI) / 180.0);
    ctx.fillRect(0, 0, 4, 8);
    ctx.restore();
  });

  ctx.restore();
}

// =====================================================================
// 4. TELEMETRY & CONTROLS BINDING
// =====================================================================
function updateCockpitInstruments(rover) {
  if (!rover) return;

  // Heading Needle & Readout
  const needle = document.getElementById("compass-needle");
  const yawVal = document.getElementById("rover-yaw-val");
  const dirVal = document.getElementById("rover-dir-val");
  if (needle) needle.style.transform = `rotate(${rover.yaw_deg}deg)`;
  if (yawVal) yawVal.textContent = `${Math.round(rover.yaw_deg).toString().padStart(3, "0")}°`;

  if (dirVal) {
    const y = rover.yaw_deg % 360;
    let d = "N";
    if (y >= 22.5 && y < 67.5) d = "NE";
    else if (y >= 67.5 && y < 112.5) d = "E";
    else if (y >= 112.5 && y < 157.5) d = "SE";
    else if (y >= 157.5 && y < 202.5) d = "S";
    else if (y >= 202.5 && y < 247.5) d = "SW";
    else if (y >= 247.5 && y < 292.5) d = "W";
    else if (y >= 292.5 && y < 337.5) d = "NW";
    dirVal.textContent = d;
  }

  // Pitch & Roll
  const pVal = document.getElementById("rover-pitch-val");
  const rVal = document.getElementById("rover-roll-val");
  const sVal = document.getElementById("rover-steer-val");
  if (pVal) pVal.textContent = `${rover.pitch_deg >= 0 ? "+" : ""}${rover.pitch_deg.toFixed(1)}°`;
  if (rVal) rVal.textContent = `${rover.roll_deg >= 0 ? "+" : ""}${rover.roll_deg.toFixed(1)}°`;
  if (sVal) sVal.textContent = `${rover.steering_deg.toFixed(1)}°`;

  // Odometry
  const posVal = document.getElementById("rover-pos-val");
  const speedVal = document.getElementById("rover-speed-val");
  if (posVal) posVal.textContent = `X: ${rover.x.toFixed(1)}m Y: ${rover.y.toFixed(1)}m`;
  if (speedVal) speedVal.textContent = `${rover.speed_mps.toFixed(1)} m/s (${(rover.speed_mps * 3.6).toFixed(1)} km/h)`;
}

function updateStageLatencies(timings) {
  if (!timings) return;
  const t1 = document.getElementById("t-stage1");
  const t2 = document.getElementById("t-stage2");
  const t3 = document.getElementById("t-stage3");
  const t4 = document.getElementById("t-stage4");

  if (t1) t1.textContent = `${(timings.stage1_spatiotemporal_ms || 0.4).toFixed(1)} ms`;
  if (t2) t2.textContent = `${(timings.stage2_segmentation_ms || 2.1).toFixed(1)} ms`;
  if (t3) t3.textContent = `${(timings.stage3_4_mapping_ms * 0.4 || 0.8).toFixed(1)} ms`;
  if (t4) t4.textContent = `${(timings.stage3_4_mapping_ms * 0.6 || 1.2).toFixed(1)} ms`;
}

// Drive Commands
async function sendRoverControl(cmd) {
  try {
    const resp = await fetch("/api/rover_control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cmd),
    });
    const res = await resp.json();
    if (res.rover) {
      state.rover = res.rover;
      updateCockpitInstruments(state.rover);
    }
  } catch (err) {
    console.error("Control error:", err);
  }
}

function bindControls() {
  document.getElementById("btn-drive-fwd")?.addEventListener("click", () => sendRoverControl({ drive_dist: 0.8 }));
  document.getElementById("btn-drive-rev")?.addEventListener("click", () => sendRoverControl({ drive_dist: -0.8 }));
  document.getElementById("btn-steer-left")?.addEventListener("click", () => sendRoverControl({ yaw_delta: -10, steering_deg: -15 }));
  document.getElementById("btn-steer-right")?.addEventListener("click", () => sendRoverControl({ yaw_delta: 10, steering_deg: 15 }));

  const patrolBtn = document.getElementById("btn-patrol");
  patrolBtn?.addEventListener("click", () => {
    state.isAutoPatrol = !state.isAutoPatrol;
    patrolBtn.classList.toggle("btn-patrol", state.isAutoPatrol);
    patrolBtn.textContent = state.isAutoPatrol ? "PATROLLING" : "PATROL";
  });

  // Keyboard Driving Controls (WASD)
  window.addEventListener("keydown", (e) => {
    if (["input", "select", "textarea"].includes(document.activeElement.tagName.toLowerCase())) return;
    if (e.key === "w" || e.key === "ArrowUp") sendRoverControl({ drive_dist: 0.6 });
    else if (e.key === "s" || e.key === "ArrowDown") sendRoverControl({ drive_dist: -0.6 });
    else if (e.key === "a" || e.key === "ArrowLeft") sendRoverControl({ yaw_delta: -8, steering_deg: -12 });
    else if (e.key === "d" || e.key === "ArrowRight") sendRoverControl({ yaw_delta: 8, steering_deg: 12 });
  });

  // Layer Selectors
  document.getElementById("select-color-mode")?.addEventListener("change", (e) => {
    state.colorMode = e.target.value;
    if (state.lastFrameData) updateThreePointCloud(state.lastFrameData.points_sample, state.colorMode);
  });

  document.getElementById("select-dogma-layer")?.addEventListener("change", (e) => {
    state.dogmaLayer = e.target.value;
    if (state.lastFrameData) drawDogma2D(state.lastFrameData);
  });

  document.getElementById("select-scenario")?.addEventListener("change", (e) => {
    state.currentScenario = e.target.value;
    state.currentFrame = 0;
  });

  document.getElementById("btn-play")?.addEventListener("click", () => { state.isPlaying = true; });
  document.getElementById("btn-pause")?.addEventListener("click", () => { state.isPlaying = false; });
  document.getElementById("btn-step")?.addEventListener("click", () => { fetchSweepFrame(); });

  // 7-Fold CV Modal
  const modal = document.getElementById("cv-modal");
  document.getElementById("btn-cv-modal")?.addEventListener("click", () => {
    modal?.classList.add("open");
    loadTrainingMetrics();
  });
  document.getElementById("btn-close-modal")?.addEventListener("click", () => {
    modal?.classList.remove("open");
  });
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("open");
  });
}

// Fetch Sweep Frame
async function fetchSweepFrame() {
  try {
    if (state.isAutoPatrol) {
      state.rover.yaw_deg = (state.rover.yaw_deg + 1.2) % 360;
      const rad = (state.rover.yaw_deg * Math.PI) / 180;
      state.rover.x += 0.08 * Math.sin(rad);
      state.rover.y += 0.08 * Math.cos(rad);
      state.rover.pitch_deg = 2.0 * Math.sin(state.currentFrame * 0.1);
      state.rover.roll_deg = 1.2 * Math.cos(state.currentFrame * 0.12);
    }

    const url = `/api/sweep?scenario=${state.currentScenario}&frame=${state.currentFrame}`;
    const resp = await fetch(url);
    const data = await resp.json();

    state.lastFrameData = data;
    const maxFrames = state.currentScenario === "eigensight_stacking" ? 4 : 50;
    state.currentFrame = (state.currentFrame + 1) % maxFrames;

    // Update Eigensight live banner matching screenshot
    const bannerEl = document.getElementById("eigensight-banner");
    if (data.meta && data.meta.frame_label) {
      if (bannerEl) bannerEl.style.display = "block";
      const frameEl = document.getElementById("banner-frame-lbl");
      if (frameEl) frameEl.textContent = data.meta.frame_label;
    } else {
      if (bannerEl) bannerEl.style.display = "none";
    }

    // Update Viewers & Telemetry
    updateThreePointCloud(data.points_sample, state.colorMode);
    drawDogma2D(data);
    updateCockpitInstruments(data.rover);
    updateStageLatencies(data.timings_ms);
  } catch (err) {
    console.error("Frame fetch error:", err);
  }
}

// Load 7,000 Pictures Training Metrics
async function loadTrainingMetrics() {
  try {
    const resp = await fetch("/api/training_metrics");
    const m = await resp.json();

    if (m.total_pictures) {
      document.getElementById("m-pictures").textContent = m.total_pictures.toLocaleString();
      document.getElementById("m-points").textContent = (m.points_evaluated || 3584000).toLocaleString();
      document.getElementById("m-val-acc").textContent = `${m.mean_val_accuracy.toFixed(2)}%`;
      document.getElementById("m-gap").textContent = m.mean_generalization_gap.toFixed(4);
      document.getElementById("m-diagnosis").textContent = m.overfitting_diagnosis;

      const tbody = document.getElementById("cv-tbody");
      if (tbody && m.fold_breakdown) {
        tbody.innerHTML = m.fold_breakdown.map(f => `
          <tr>
            <td>Fold ${f.fold}</td>
            <td>${f.train_loss.toFixed(4)}</td>
            <td>${f.val_loss.toFixed(4)}</td>
            <td>${f.train_acc.toFixed(2)}%</td>
            <td style="color:var(--emerald);font-weight:600;">${f.val_acc.toFixed(2)}%</td>
            <td style="color:var(--dynamic);">${f.generalization_gap.toFixed(4)}</td>
          </tr>
        `).join("");
      }
    }
  } catch (err) {
    console.error("Metrics fetch error:", err);
  }
}

// =====================================================================
// 5. INTERACTIVE 2.5D WELFORD & NON-UNIFORM GRID ANIMATIONS
// =====================================================================
function initWelfordAnimation() {
  const canvas = document.getElementById("welford-anim-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const w = canvas.width;
  const h = canvas.height;

  // Stream of simulated points dropping into 5cm micro-cell
  const points = [];
  const maxPts = 28;

  function animLoop() {
    ctx.fillStyle = "#070A0E";
    ctx.fillRect(0, 0, w, h);

    // 1. Draw 3D Isometric 5cm Grid Voxel Box
    const cx = w / 2;
    const cy = h / 2 + 10;
    const bw = 110;
    const bh = 55;
    const depth = 35;

    // Bottom isometric base
    ctx.strokeStyle = "rgba(71, 215, 227, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy - depth);
    ctx.lineTo(cx + bw / 2, cy);
    ctx.lineTo(cx, cy + depth);
    ctx.lineTo(cx - bw / 2, cy);
    ctx.closePath();
    ctx.stroke();

    // Top bounding ring
    const topOffset = 45;
    ctx.strokeStyle = "rgba(71, 215, 227, 0.2)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, cy - depth - topOffset);
    ctx.lineTo(cx + bw / 2, cy - topOffset);
    ctx.lineTo(cx, cy + depth - topOffset);
    ctx.lineTo(cx - bw / 2, cy - topOffset);
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

    // Vertical corner pillars
    ctx.strokeStyle = "rgba(71, 215, 227, 0.25)";
    [
      [cx, cy - depth],
      [cx + bw / 2, cy],
      [cx, cy + depth],
      [cx - bw / 2, cy],
    ].forEach(([px, py]) => {
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px, py - topOffset);
      ctx.stroke();
    });

    // Spawn falling points
    if (Math.random() < 0.35 && points.length < maxPts) {
      // Simulate curb height jump
      const isCurbPoint = Math.random() < 0.4;
      const targetZ = isCurbPoint ? 0.12 + Math.random() * 0.08 : -0.02 + Math.random() * 0.04;
      const isoX = (Math.random() - 0.5) * (bw * 0.7);
      const isoY = (Math.random() - 0.5) * (depth * 0.7);
      points.push({
        x: cx + isoX,
        y: -10,
        targetY: cy + isoY - targetZ * 120,
        z: targetZ,
        radius: 2.2,
        color: isCurbPoint ? "#F59E0B" : "#34D399",
      });
    }

    // Update and draw points
    let sumZ = 0;
    let sumZ2 = 0;
    const landed = [];

    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i];
      if (p.y < p.targetY) {
        p.y += 3.5;
        // Motion spark
        ctx.fillStyle = "rgba(71, 215, 227, 0.3)";
        ctx.fillRect(p.x - 0.5, p.y - 6, 1, 6);
      } else {
        landed.push(p);
        sumZ += p.z;
        sumZ2 += p.z * p.z;
      }

      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Keep point count managed
    if (landed.length > 24) {
      points.shift();
    }

    // Live Math Calculations
    const n = landed.length;
    if (n >= 2) {
      const meanZ = sumZ / n;
      const varZ = Math.max(0, (sumZ2 - (sumZ * sumZ) / n) / (n - 1));
      const stdZ = Math.sqrt(varZ);

      // Mean elevation plane
      const meanY = cy - meanZ * 120;
      ctx.strokeStyle = "rgba(71, 215, 227, 0.7)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 2]);
      ctx.beginPath();
      ctx.moveTo(cx - bw / 3, meanY);
      ctx.lineTo(cx + bw / 3, meanY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(71, 215, 227, 0.9)";
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.fillText(`μ = ${(meanZ > 0 ? "+" : "") + meanZ.toFixed(3)}m`, cx + bw / 3 + 4, meanY + 3);

      // Update Card 1 UI Readouts
      const nEl = document.getElementById("calc-n-samples");
      const mEl = document.getElementById("calc-mean-z");
      const vEl = document.getElementById("calc-var-z");
      const sEl = document.getElementById("calc-std-z");
      const bEl = document.getElementById("calc-terrain-badge");

      if (nEl) nEl.textContent = `${n} pts`;
      if (mEl) mEl.textContent = `${(meanZ > 0 ? "+" : "") + meanZ.toFixed(3)} m`;
      if (vEl) vEl.textContent = `${varZ.toFixed(4)} m²`;
      if (sEl) sEl.textContent = `${stdZ.toFixed(3)} m`;

      if (bEl) {
        if (stdZ >= 0.08) {
          bEl.className = "badge-status status-hazard";
          bEl.textContent = "CURB HAZARD (σz ≥ 0.08m)";
        } else if (stdZ >= 0.04) {
          bEl.className = "badge-status";
          bEl.style.color = "#E8A34D";
          bEl.style.borderColor = "#E8A34D";
          bEl.textContent = "MILD UNDULATION";
        } else {
          bEl.className = "badge-status status-saving";
          bEl.textContent = "SMOOTH GROUND (σz < 0.04m)";
        }
      }
    }

    requestAnimationFrame(animLoop);
  }
  animLoop();
}

function initNonUniformAnimation() {
  const canvas = document.getElementById("nonuniform-anim-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const w = canvas.width;
  const h = canvas.height;

  let scanAngle = 0;

  function animLoop() {
    ctx.fillStyle = "#070A0E";
    ctx.fillRect(0, 0, w, h);

    const pad = 12;
    const gw = w - pad * 2;
    const gh = h - pad * 2;
    const x0 = pad;
    const y0 = pad;

    // 9 Partitions: 3x3 layout (Buerkle et al. Fig. 2)
    // Band widths: 30%, 40%, 30%
    const colW = [gw * 0.30, gw * 0.40, gw * 0.30];
    const rowH = [gh * 0.30, gh * 0.40, gh * 0.30];

    const colX = [x0, x0 + colW[0], x0 + colW[0] + colW[1]];
    const rowY = [y0, y0 + rowH[0], y0 + rowH[0] + rowH[1]];

    // Cell size labels per partition (Table II)
    const partInfo = [
      { name: "X0", res: "0.4m", cells: 4 },
      { name: "X1", res: "0.2m", cells: 8 },
      { name: "X2", res: "0.4m", cells: 4 },
      { name: "X3", res: "0.2m", cells: 8 },
      { name: "X4", res: "0.1m", cells: 16, center: true },
      { name: "X5", res: "0.2m", cells: 8 },
      { name: "X6", res: "0.4m", cells: 4 },
      { name: "X7", res: "0.2m", cells: 8 },
      { name: "X8", res: "0.4m", cells: 4 },
    ];

    // Draw Partitions
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const idx = r * 3 + c;
        const info = partInfo[idx];
        const px = colX[c];
        const py = rowY[r];
        const pw = colW[c];
        const ph = rowH[r];

        // Background tint
        if (info.center) {
          ctx.fillStyle = "rgba(71, 215, 227, 0.08)";
        } else if (info.res === "0.2m") {
          ctx.fillStyle = "rgba(168, 85, 247, 0.05)";
        } else {
          ctx.fillStyle = "rgba(30, 41, 59, 0.3)";
        }
        ctx.fillRect(px, py, pw, ph);

        // Partition Border
        ctx.strokeStyle = info.center ? "rgba(71, 215, 227, 0.65)" : "rgba(100, 116, 139, 0.35)";
        ctx.lineWidth = info.center ? 1.5 : 1;
        ctx.strokeRect(px, py, pw, ph);

        // Internal cell grid lines showing relative resolution
        const nSub = info.cells;
        const subW = pw / nSub;
        const subH = ph / nSub;
        ctx.strokeStyle = info.center ? "rgba(71, 215, 227, 0.18)" : "rgba(100, 116, 139, 0.12)";
        ctx.lineWidth = 0.5;

        for (let sx = 1; sx < nSub; sx++) {
          ctx.beginPath();
          ctx.moveTo(px + sx * subW, py);
          ctx.lineTo(px + sx * subW, py + ph);
          ctx.stroke();
        }
        for (let sy = 1; sy < nSub; sy++) {
          ctx.beginPath();
          ctx.moveTo(px, py + sy * subH);
          ctx.lineTo(px + pw, py + sy * subH);
          ctx.stroke();
        }

        // Labels
        ctx.fillStyle = info.center ? "rgba(71, 215, 227, 0.95)" : "rgba(148, 163, 184, 0.7)";
        ctx.font = info.center ? "600 10px 'IBM Plex Mono', monospace" : "9px 'IBM Plex Mono', monospace";
        ctx.fillText(`${info.name} (${info.res})`, px + 4, py + 12);
      }
    }

    // Dynamic obstacle representation (Cyan Car moving through center)
    const t = (Date.now() % 4000) / 4000;
    const carX = colX[1] + colW[1] / 2 + Math.sin(t * Math.PI * 2) * (colW[1] * 0.28);
    const carY = rowY[1] + colH[1] / 2 + (t - 0.5) * (colH[1] * 0.6);

    ctx.fillStyle = "rgba(59, 130, 246, 0.9)"; // Blue (Dynamic per paper)
    ctx.shadowColor = "#3B82F6";
    ctx.shadowBlur = 8;
    ctx.fillRect(carX - 5, carY - 8, 10, 16);
    ctx.shadowBlur = 0;

    // Static obstacles (Red per paper Fig. 8)
    ctx.fillStyle = "rgba(239, 68, 68, 0.85)";
    ctx.fillRect(colX[1] + colW[1] * 0.65, rowY[1] + 6, 8, 14); // Parked Red car
    ctx.fillRect(colX[1] + colW[1] * 0.25, rowY[1] + rowH[1] * 0.6, 10, 10); // Tree

    // Sweep scan line
    scanAngle = (scanAngle + 0.02) % (Math.PI * 2);
    const lineX = x0 + (Math.sin(scanAngle) * 0.5 + 0.5) * gw;
    ctx.strokeStyle = "rgba(71, 215, 227, 0.35)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(lineX, y0);
    ctx.lineTo(lineX, y0 + gh);
    ctx.stroke();

    requestAnimationFrame(animLoop);
  }
  animLoop();
}

// Bind Model Switcher
function bindModelToggle() {
  const btnTransfer = document.getElementById("btn-model-transfer");
  const btnSparse = document.getElementById("btn-model-sparse");

  async function setMode(mode) {
    try {
      await fetch("/api/model_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      if (mode === "transfer_learning") {
        btnTransfer?.classList.add("active");
        btnSparse?.classList.remove("active");
      } else {
        btnSparse?.classList.add("active");
        btnTransfer?.classList.remove("active");
      }
    } catch (e) {
      console.error("Model mode switch error:", e);
    }
  }

  btnTransfer?.addEventListener("click", () => setMode("transfer_learning"));
  btnSparse?.addEventListener("click", () => setMode("sparse_cnn"));
}

// App Loop
function startAppLoop() {
  setInterval(() => {
    if (state.isPlaying) {
      fetchSweepFrame();
    }
  }, 120);
}

// Initialization
window.addEventListener("DOMContentLoaded", () => {
  initHeroCanvas();
  initThreeJS();
  initDogmaCanvas();
  initWelfordAnimation();
  initNonUniformAnimation();
  bindModelToggle();
  bindControls();
  fetchSweepFrame();
  startAppLoop();
});
