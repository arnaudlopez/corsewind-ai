const DATA_VERSION = new URLSearchParams(window.location.search).get("v") || String(Date.now());
const RASTER_ENABLED = new URLSearchParams(window.location.search).get("raster") !== "0";
const versionedDataUrl = (url) => `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(DATA_VERSION)}`;
const cacheBustedUrl = (url) => `${url}${url.includes("?") ? "&" : "?"}poll=${Date.now()}`;
const gzipJsonUrl = (url) => url.replace(/\.json(?=([?#]|$))/, ".json.gz");
const MODEL_UPDATE_POLL_INTERVAL_MS = 60_000;
const AROME_DATA_URL = versionedDataUrl("./arome-corsica-latest.json");
const AROMEPI_DATA_URL = versionedDataUrl("./aromepi-corsica-latest.json");
const MOLOCH_DATA_URL = versionedDataUrl("./moloch-corsica-latest.json");
const ICON2I_DATA_URL = versionedDataUrl("./icon2i-corsica-latest.json");
const INITIAL_MODEL_URLS = [
  { layer: "aromepi", url: AROMEPI_DATA_URL },
  { layer: "arome", url: AROME_DATA_URL },
  { layer: "icon2i", url: ICON2I_DATA_URL },
  { layer: "moloch", url: MOLOCH_DATA_URL },
];
const RASTER_TILES_MANIFEST_URL = versionedDataUrl("./tiles/manifest.json");
const modelRasterManifestUrl = (model) => versionedDataUrl(`./tiles/${model}/manifest.json`);
const WINDNINJA_CORSICA_50M_DATA_MANIFEST_URL = versionedDataUrl("./windninja-corsica-data-50m/manifest.json");
const WINDNINJA_CORSICA_50M_TILES_MANIFEST_URL = versionedDataUrl("./windninja-corsica-tiles-50m/manifest.json");
const CFD_URL = versionedDataUrl("../../data/processed/physics/ajaccio_cfd_pilot/cfd_micro50_smoke_layer.json");
const COASTAL_TILES_URL = versionedDataUrl("../../data/processed/physics/coastal_cfd_tile_plan.json");
const BAY_MODEL_URL = versionedDataUrl("../../data/processed/physics/ajaccio_bay_1m_model_plan.json");
const MULTISCALE_PLAN_URL = versionedDataUrl("../../data/processed/physics/ajaccio_multiscale_domain_plan.json");
const EXPANDED_WIND_5KM_URL = versionedDataUrl("../../data/processed/physics/ajaccio_expanded_5km_wind_overview.json");
const PRIORITY_CORRIDORS_URL = versionedDataUrl("../../data/processed/physics/ajaccio_priority_corridor_tiles.json");
const LOCAL_WIND_2M_URL = versionedDataUrl("../../data/processed/physics/ajaccio_local_wind_2m_grid.json");
const SPOT_GRIDS_URL = versionedDataUrl("../../data/processed/physics/ajaccio_windsurf_spot_grids.json");
const WINDNINJA_SPOT_GRIDS_URL = versionedDataUrl("../../data/processed/physics/ajaccio_windninja_spot_grids.json");
const REGIME_QA_URL = versionedDataUrl("../../data/processed/physics/ajaccio_windsurf_regime_qa.json");
const VALIDATION_URL = versionedDataUrl("../../data/processed/validation/ajaccio_windsurf_casebook.json");
const VALIDATION_GAPS_URL = versionedDataUrl("../../data/processed/validation/ajaccio_windsurf_session_decision_gaps.json");
const FIELD_TEST_PACKET_URL = versionedDataUrl("../../data/processed/validation/ajaccio_windsurf_field_test_packet.json");
const BASEMAP_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const CENTER = [42.14, 9.08];
const VIEW_BOUNDS = [
  [41.25, 8.45],
  [43.1, 9.75],
];
const INITIAL_ZOOM = 8;
const KNOTS_PER_MPS = 1.943844492;
const DEFAULT_SCALE_MAX_KT = 14;
const DISPLAY_TIME_ZONE = "Europe/Paris";
const CFD_INFLUENCE_RADIUS_M = 260;
const CFD_SIGMA_M = 105;
const SESSION_CLASSES = {
  0: { key: "too_light", label: "Trop léger", rgb: [37, 99, 235], priority: 0.44 },
  1: { key: "offshore_caution", label: "Offshore prudent", rgb: [245, 139, 42], priority: 0.74 },
  2: { key: "devente", label: "Dévente", rgb: [248, 250, 252], priority: 0.86 },
  3: { key: "gusty", label: "Rafaleux", rgb: [239, 68, 68], priority: 0.92 },
  4: { key: "accelerated", label: "Accéléré", rgb: [250, 204, 21], priority: 0.82 },
  5: { key: "clean", label: "Propre", rgb: [34, 197, 94], priority: 0.72 },
  6: { key: "marginal", label: "Marginal", rgb: [20, 184, 166], priority: 0.56 },
  7: { key: "low_confidence", label: "Confiance faible", rgb: [100, 116, 139], priority: 0.38 },
};
const SESSION_LEGEND_ORDER = [5, 4, 6, 0, 1, 2, 3, 7];
const SURFACE_CLASSES = {
  0: { label: "Source faible", rgb: [71, 85, 105] },
  1: { label: "Mer / eau", rgb: [37, 99, 235] },
  2: { label: "Bande côtière", rgb: [250, 204, 21] },
  3: { label: "Urbain / obstacle", rgb: [239, 68, 68] },
  4: { label: "Relief exposé", rgb: [168, 85, 247] },
  5: { label: "Terre peu rugueuse", rgb: [34, 197, 94] },
};

class AromeWindOverlay extends L.Layer {
  constructor(payload, cfdPayload = null, coastalTilePayload = null, bayModelPayload = null, multiscalePlanPayload = null, expandedWindPayload = null, localWindPayload = null, spotGridPayload = null, windNinjaSpotPayload = null, validationPayload = null, regimeQaPayload = null, validationGapsPayload = null, fieldTestPacketPayload = null, rasterTilePayload = null, windNinjaCorsicaTilePayload = null, windNinjaCorsica1mTilePayload = null, windNinjaCorsica50mTilePayload = null, molochPayload = null, icon2iPayload = null, aromePiPayload = null) {
    super();
    this.payload = payload;
    this.bbox = payload.bbox_wgs84;
    this.primaryRawLayer = rawLayerKeyForPayload(payload) || "arome";
    this.arome = this.primaryRawLayer === "arome" ? buildRawWindLayer(payload) : null;
    this.moloch = buildRawWindLayer(molochPayload || (this.primaryRawLayer === "moloch" ? payload : null));
    this.icon2i = buildRawWindLayer(icon2iPayload || (this.primaryRawLayer === "icon2i" ? payload : null));
    this.aromepi = buildRawWindLayer(aromePiPayload || (this.primaryRawLayer === "aromepi" ? payload : null));
    this.cfd = buildCfdCorrection(cfdPayload);
    this.coastalTiles = buildCoastalTileLayer(coastalTilePayload);
    this.bayModel = buildBayModelLayer(bayModelPayload);
    this.multiscalePlan = buildMultiscalePlanLayer(multiscalePlanPayload);
    this.expandedWind = buildExpandedWindLayer(expandedWindPayload);
    this.priorityCorridorManifest = null;
    this.priorityCorridorManifestPromise = null;
    this.priorityCorridorLoadingIds = new Set();
    this.priorityCorridors = { corridors: [] };
    this.localWind = buildLocalWindLayer(localWindPayload);
    this.spotGrids = buildSpotGridLayer(spotGridPayload);
    this.windNinjaSpots = buildSpotGridLayer(windNinjaSpotPayload);
    this.validation = buildValidationState(validationPayload);
    this.regimeQa = buildRegimeQaState(regimeQaPayload);
    this.validationGaps = buildValidationGapState(validationGapsPayload);
    this.fieldTestPacket = buildFieldTestPacketState(fieldTestPacketPayload);
    // Pre-baked colour tiles per raw model (arome/aromepi/moloch/icon2i). rasterTiles always
    // points at the active model's state; rasterTilesByModel holds every loaded manifest.
    this.rasterTiles = buildRasterTileState(rasterTilePayload);
    this.rasterTilesByModel = {};
    this.windNinjaCorsicaTiles = buildRasterTileState(windNinjaCorsicaTilePayload);
    this.windNinjaCorsica1mTiles = buildRasterTileState(windNinjaCorsica1mTilePayload);
    this.windNinjaCorsica50mTiles = buildRasterTileState(windNinjaCorsica50mTilePayload);
    this.rawLayerLoading = { arome: false, aromepi: false, moloch: false, icon2i: false };
    this.rawLayerLoadError = { arome: null, aromepi: null, moloch: null, icon2i: null };
    this.rawLayerLoadPromises = {};
    this.rawLayerResourceSignatures = {};
    this.rawLayerPayloadSignatures = {};
    this.visibleLayers = { arome: this.primaryRawLayer === "arome", aromepi: this.primaryRawLayer === "aromepi", moloch: false, icon2i: false, windninja50: false };
    this.displayMode = "speed";
    this.stepIndex = 0;
    this.activeLeadHour = Number(payload.forecast_steps[0]?.lead_hour ?? 0);
    this.scaleMaxKnots = DEFAULT_SCALE_MAX_KT;
    this.particlesEnabled = true;
    this.particleOpacity = 3;
    this.particleDensity = 3;
    this.particleLifeScale = 4;
    this.particleSizeScale = 1.5;
    this.particles = [];
    this.particleFrame = null;
    this.lastParticleDrawTime = null;
    this.lastParticleTime = null;
    this.windNinjaDataTileCache = new Map();
    this.heatTileRenderCache = new Map();
    this.heatTilePrewarmKeys = new Set();
    this.heatTilePrewarmIdle = null;
    this.heatTileLayer = null;
    this.heatCanvas = document.createElement("canvas");
    this.heatCtx = this.heatCanvas.getContext("2d", { alpha: true });
    this.zoomAnimating = false;
    this.zoomAnimationState = null;
    this.zoomAnimationMode = null;
    this.zoomResetFrame = null;
    this.zoomResetTimer = null;
    this.zoomResetIdle = null;
    this.tilePrefetchCache = new Map();
    this.tilePrefetchFrame = null;
    this.lastTilePrefetchKey = null;
    this.inputPauseUntil = 0;
    this.inputPauseTimer = null;
    this.scaleRedrawTimer = null;
    this.motionActive = false;
    this.motionShowTimer = null;
    this.renderView = null;
    this.renderMetrics = null;
    this.shell = null;
  }

  get step() {
    return forecastStepByLead(this.payload, this.activeLeadHour);
  }

  get aromeStep() {
    return forecastStepByLead(this.arome, this.activeLeadHour);
  }

  get localStep() {
    return this.localWind?.forecast_steps?.[this.stepIndex] || null;
  }

  get molochStep() {
    return forecastStepByLead(this.moloch, this.activeLeadHour);
  }

  get icon2iStep() {
    return forecastStepByLead(this.icon2i, this.activeLeadHour);
  }

  get aromePiStep() {
    return forecastStepByLead(this.aromepi, this.activeLeadHour);
  }

  get expandedStep() {
    return this.expandedWind?.forecast_steps?.[this.stepIndex] || null;
  }

  setStep(index, leadHour = null) {
    this.stepIndex = index;
    this.activeLeadHour = Number(leadHour ?? this.payload.forecast_steps[index]?.lead_hour ?? this.activeLeadHour);
    applyPreferredForecastLayer(this);
    this.heatDirty = true;
    // No clearHeatTileCache(): the lead hour is part of the cache key, so switching steps reuses
    // previously rendered tiles instead of re-rasterising them. The LRU handles eviction.
    this.windNinjaDataTileCache.clear();
    this.resetParticles();
    this.refreshTileLayers();
    this.redrawHeatLayer();
    refreshPinnedPointInspector(this);
  }

  setScaleMaxKnots(value) {
    this.scaleMaxKnots = Math.max(6, Math.min(80, Number(value) || DEFAULT_SCALE_MAX_KT));
    this.heatDirty = true;
    // The scale value is part of the heat-tile cache key, so we never need to wipe the cache:
    // each scale produces distinct keys and the LRU evicts old ones. Debounce the heavy redraw
    // so dragging the slider doesn't re-render every tile on each input event.
    if (this.scaleRedrawTimer) window.clearTimeout(this.scaleRedrawTimer);
    this.scaleRedrawTimer = window.setTimeout(() => {
      this.scaleRedrawTimer = null;
      this.redrawWindNinjaDataLayers();
      this.redrawHeatLayer();
    }, 140);
  }

  setDisplayMode(mode) {
    const nextMode = ["speed", "devente", "acceleration"].includes(mode) ? mode : "speed";
    this.displayMode = windNinjaModesAvailable(this) ? nextMode : "speed";
    this.heatDirty = true;
    this.clearHeatTileCache();
    this.syncCanvasVisibility();
    this.syncParticleVisibility();
    this.refreshTileLayers();
    syncModeControls(this);
    updateLegendTitle(this.displayMode);
    this.redrawHeatLayer();
  }

  setLayerVisible(layer, visible) {
    if (!["arome", "aromepi", "moloch", "icon2i", "windninja50"].includes(layer)) return;
    if (layer === "arome" && !this.arome) return;
    if (layer === "aromepi" && !this.aromepi) return;
    if (layer === "moloch" && !this.moloch) return;
    if (layer === "icon2i" && !this.icon2i) return;
    if (visible) {
      const equivalentStep = equivalentStepForLayer(this, layer);
      if (!equivalentStep) return;
      setActiveLeadHour(this, equivalentStep.lead_hour);
    }
    if (visible && isRawLayerKey(layer)) {
      for (const rawLayer of RAW_LAYER_KEYS) this.visibleLayers[rawLayer] = rawLayer === layer;
      this.visibleLayers.windninja50 = false;
    }
    if (visible && layer === "windninja50") {
      for (const rawLayer of RAW_LAYER_KEYS) this.visibleLayers[rawLayer] = false;
    }
    this.visibleLayers[layer] = Boolean(visible);
    if (!anyRawLayerVisible(this) && !this.visibleLayers.windninja50) setFirstAvailableRawLayerVisible(this);
    this.syncCanvasVisibility();
    this.syncParticleVisibility();
    this.heatDirty = true;
    // No clearHeatTileCache(): the active model is part of the cache key, so toggling layers
    // reuses tiles already rendered for each model rather than re-rasterising them.
    this.windNinjaDataTileCache.clear();
    this.resetParticles();
    this.refreshTileLayers();
    this.redrawHeatLayer();
    refreshActiveLayerLabel(this);
    refreshCoverageStatus(this);
    syncModeControls(this);
    refreshPinnedPointInspector(this);
  }

  setParticlesEnabled(enabled) {
    this.particlesEnabled = Boolean(enabled);
    this.syncParticleVisibility();
    this.resetParticles();
  }

  setParticleOpacity(value) {
    this.particleOpacity = Math.max(0.1, Math.min(5, Number(value) || 1));
  }

  setParticleDensity(value) {
    this.particleDensity = Math.max(0.2, Math.min(5, Number(value) || 1));
    this.resetParticles();
  }

  setParticleLifeScale(value) {
    this.particleLifeScale = Math.max(0.2, Math.min(5, Number(value) || 1));
    this.resetParticles();
  }

  setParticleSizeScale(value) {
    this.particleSizeScale = Math.max(0.25, Math.min(5, Number(value) || 1));
  }

  refreshTileLayers() {
    updateRasterTileLayer(this);
    updateWindNinjaCorsicaTileLayer(this);
    // Raster availability can change with zoom/step/model (e.g. crossing the native min zoom),
    // so re-sync the JS heat layer: hide it when raster covers, restore + redraw it otherwise.
    this.redrawHeatLayer();
  }

  heatLayerVisible() {
    if (!anyRawLayerVisible(this) || this.displayMode !== "speed") return false;
    // When pre-baked raster tiles cover the active model/step/zoom, they replace the JS heat
    // layer entirely — drawing both would double the overlay and waste CPU.
    if (activeModelRasterAvailable(this)) return false;
    return true;
  }

  syncHeatTileLayerVisibility() {
    if (!this.heatTileLayer) return;
    const container = this.heatTileLayer.getContainer?.();
    if (container) container.style.display = this.heatLayerVisible() ? "" : "none";
  }

  redrawHeatLayer() {
    if (!this.heatTileLayer) return;
    this.syncHeatTileLayerVisibility();
    if (this.heatLayerVisible()) this.heatTileLayer.redraw();
  }

  clearHeatTileCache() {
    this.heatTileRenderCache.clear();
    this.heatTilePrewarmKeys.clear();
    if (this.heatTilePrewarmIdle && window.cancelIdleCallback) {
      window.cancelIdleCallback(this.heatTilePrewarmIdle);
      this.heatTilePrewarmIdle = null;
    }
  }

  redrawWindNinjaDataLayers() {
    for (const tileState of [this.windNinjaCorsica50mTiles]) {
      if (tileState?.encoding === "data" && tileState.activeLayer?.redraw) {
        tileState.activeLayer.redraw();
      }
    }
  }

  syncCanvasVisibility() {
    if (!this.canvas) return;
    this.canvas.hidden = true;
  }

  syncParticleVisibility() {
    if (!this.particleCanvas) return;
    const hasWindLayer = anyRawLayerVisible(this) || this.visibleLayers.windninja50;
    this.particleCanvas.hidden = !this.particlesEnabled || !hasWindLayer;
  }

  onAdd(map) {
    this.map = map;
    this.shell = document.querySelector(".app-shell");
    if (!map.getPane("windHeatPane")) {
      const pane = map.createPane("windHeatPane");
      pane.style.zIndex = 420;
      pane.style.pointerEvents = "none";
    }
    map.getPane("windHeatPane")?.classList.add("wind-heat-pane");
    this.heatTileLayer = createWindHeatTileLayer(this);
    this.heatTileLayer.addTo(map);
    this.syncHeatTileLayerVisibility();
    this.canvas = L.DomUtil.create("canvas", "cfd-wind-canvas leaflet-zoom-animated");
    this.ctx = this.canvas.getContext("2d", { alpha: true });
    this.syncCanvasVisibility();
    this.shell.appendChild(this.canvas);
    this.particleCanvas = L.DomUtil.create("canvas", "wind-particle-canvas leaflet-zoom-animated");
    this.particleCtx = this.particleCanvas.getContext("2d", { alpha: true });
    this.syncParticleVisibility();
    this.shell.appendChild(this.particleCanvas);
    map.on("move resize", this.reset, this);
    map.on("movestart zoomstart", this.onMotionStart, this);
    map.on("moveend zoomend", this.onMotionEnd, this);
    map.on("zoomstart", this.onZoomStart, this);
    map.on("zoomanim", this.onZoomAnim, this);
    map.on("zoom", this.onZoomProgress, this);
    map.on("zoomend", this.onZoomEnd, this);
    map.on("zoomend", this.refreshTileLayers, this);
    map.on("moveend", this.onMoveEnd, this);
    map.on("zoomend moveend", this.maybeLoadPriorityCorridors, this);
    this.reset();
    this.refreshTileLayers();
    this.draw();
    this.maybeLoadPriorityCorridors();
    this.scheduleHeatTilePrewarm(map.getCenter(), map.getZoom());
    this.startParticleLoop();
  }

  onRemove(map) {
    map.off("move resize", this.reset, this);
    map.off("movestart zoomstart", this.onMotionStart, this);
    map.off("moveend zoomend", this.onMotionEnd, this);
    map.off("zoomstart", this.onZoomStart, this);
    map.off("zoomanim", this.onZoomAnim, this);
    map.off("zoom", this.onZoomProgress, this);
    map.off("zoomend", this.onZoomEnd, this);
    map.off("zoomend", this.refreshTileLayers, this);
    map.off("moveend", this.onMoveEnd, this);
    map.off("zoomend moveend", this.maybeLoadPriorityCorridors, this);
    if (this.particleFrame) cancelAnimationFrame(this.particleFrame);
    if (this.zoomResetFrame) cancelAnimationFrame(this.zoomResetFrame);
    if (this.zoomResetTimer) window.clearTimeout(this.zoomResetTimer);
    if (this.zoomResetIdle && window.cancelIdleCallback) {
      window.cancelIdleCallback(this.zoomResetIdle);
      this.zoomResetIdle = null;
    }
    if (this.tilePrefetchFrame) cancelAnimationFrame(this.tilePrefetchFrame);
    if (this.heatTilePrewarmIdle && window.cancelIdleCallback) {
      window.cancelIdleCallback(this.heatTilePrewarmIdle);
      this.heatTilePrewarmIdle = null;
    }
    if (this.inputPauseTimer) window.clearTimeout(this.inputPauseTimer);
    if (this.scaleRedrawTimer) window.clearTimeout(this.scaleRedrawTimer);
    if (this.motionShowTimer) window.clearTimeout(this.motionShowTimer);
    for (const state of Object.values(this.rasterTilesByModel || {})) {
      if (state.activeLayer) map.removeLayer(state.activeLayer);
    }
    if (this.heatTileLayer) map.removeLayer(this.heatTileLayer);
    if (this.windNinjaCorsica50mTiles?.activeLayer) map.removeLayer(this.windNinjaCorsica50mTiles.activeLayer);
    if (this.windNinjaCorsicaTiles?.activeLayer) map.removeLayer(this.windNinjaCorsicaTiles.activeLayer);
    if (this.windNinjaCorsica1mTiles?.activeLayer) map.removeLayer(this.windNinjaCorsica1mTiles.activeLayer);
    L.DomUtil.remove(this.canvas);
    L.DomUtil.remove(this.particleCanvas);
  }

  setOverlayTransform(offset, scale) {
    if (this.canvas) L.DomUtil.setTransform(this.canvas, offset, scale);
    if (this.particleCanvas) L.DomUtil.setTransform(this.particleCanvas, offset, scale);
  }

  canvasPixelRatio() {
    return Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  }

  overlayRenderMetrics(size = this.map?.getSize()) {
    if (!size) return null;
    const pad = L.point(0, 0);
    return {
      size,
      pad,
      cssSize: size,
      pixelRatio: this.canvasPixelRatio(),
    };
  }

  captureRenderView(metrics = this.renderMetrics || this.overlayRenderMetrics()) {
    if (!this.map || !metrics) return null;
    this.renderView = {
      center: this.map.getCenter(),
      zoom: this.map.getZoom(),
      size: metrics.size,
      pad: metrics.pad,
      cssSize: metrics.cssSize,
    };
    return this.renderView;
  }

  beginZoomTracking(mode = null) {
    if (!this.map) return;
    if (this.zoomResetFrame) {
      cancelAnimationFrame(this.zoomResetFrame);
      this.zoomResetFrame = null;
    }
    if (this.zoomResetTimer) {
      window.clearTimeout(this.zoomResetTimer);
      this.zoomResetTimer = null;
    }
    if (this.zoomResetIdle && window.cancelIdleCallback) {
      window.cancelIdleCallback(this.zoomResetIdle);
      this.zoomResetIdle = null;
    }
    this.zoomAnimating = true;
    this.zoomAnimationMode = mode;
    this.zoomAnimationState = this.renderView || this.captureRenderView();
    this.shell?.classList.add("wind-overlay-zooming");
    this.shell?.classList.toggle("wind-overlay-zoom-transition", mode === "animated");
    this.setOverlayTransform(L.point(0, 0), 1);
  }

  onMotionStart() {
    if (this.motionShowTimer) {
      window.clearTimeout(this.motionShowTimer);
      this.motionShowTimer = null;
    }
    if (!this.motionActive) {
      this.motionActive = true;
      this.shell?.classList.add("wind-overlay-motion");
    }
  }

  onMotionEnd() {
    // Resume particles a beat after the map settles, so continuous gestures (and touch inertia,
    // which keeps firing move events) don't flicker the canvas on/off mid-interaction.
    if (this.motionShowTimer) window.clearTimeout(this.motionShowTimer);
    this.motionShowTimer = window.setTimeout(() => {
      this.motionShowTimer = null;
      this.motionActive = false;
      this.shell?.classList.remove("wind-overlay-motion");
      this.lastParticleDrawTime = null;
      this.lastParticleTime = null;
      // Reseed particles for the settled view (skipped during the gesture while hidden).
      if (this.particleCanvas && !this.particleCanvas.hidden) this.resetParticles();
    }, 200);
  }

  prepareZoomInput(durationMs = 130) {
    this.inputPauseUntil = performance.now() + durationMs;
    this.lastParticleDrawTime = null;
    this.lastParticleTime = null;
    this.shell?.classList.add("wind-overlay-input-pending");
    if (this.inputPauseTimer) window.clearTimeout(this.inputPauseTimer);
    this.inputPauseTimer = window.setTimeout(() => {
      this.inputPauseTimer = null;
      this.shell?.classList.remove("wind-overlay-input-pending");
    }, durationMs);
  }

  onZoomStart() {
    this.beginZoomTracking(null);
    this.prefetchBasemapTiles(this.map.getCenter(), this.map.getZoom());
    this.scheduleHeatTilePrewarm(this.map.getCenter(), this.map.getZoom());
  }

  applyZoomTransform(center, zoom) {
    if (!this.map) return;
    if (!this.zoomAnimationState) this.beginZoomTracking(null);
    const state = this.zoomAnimationState;
    if (!state) return;
    const size = state.size || this.map.getSize();
    const pad = state.pad || L.point(0, 0);
    const targetCenter = center || this.map.getCenter();
    const targetZoom = zoom ?? this.map.getZoom();
    const scale = this.map.getZoomScale(targetZoom, state.zoom);
    const startTopLeft = this.map.project(state.center, state.zoom).subtract(size.divideBy(2)).subtract(pad);
    const targetTopLeft = this.map.project(targetCenter, targetZoom).subtract(size.divideBy(2)).subtract(pad);
    const offset = startTopLeft.multiplyBy(scale).subtract(targetTopLeft);
    this.setOverlayTransform(offset, scale);
  }

  onZoomAnim(event) {
    if (!this.map) return;
    if (!this.zoomAnimating || !this.zoomAnimationState) this.beginZoomTracking("animated");
    this.zoomAnimationMode = "animated";
    this.shell?.classList.add("wind-overlay-zoom-transition");
    this.applyZoomTransform(event.center, event.zoom);
    this.prefetchBasemapTiles(event.center, event.zoom);
    this.scheduleHeatTilePrewarm(event.center, event.zoom);
  }

  onZoomProgress() {
    if (!this.map || this.zoomAnimationMode === "animated") return;
    if (!this.zoomAnimating || !this.zoomAnimationState) this.beginZoomTracking(null);
    this.shell?.classList.remove("wind-overlay-zoom-transition");
    this.applyZoomTransform(this.map.getCenter(), this.map.getZoom());
    this.prefetchBasemapTiles(this.map.getCenter(), this.map.getZoom());
    this.scheduleHeatTilePrewarm(this.map.getCenter(), this.map.getZoom());
  }

  onMoveEnd() {
    if (!this.map || this.zoomAnimating) return;
    this.scheduleHeatTilePrewarm(this.map.getCenter(), this.map.getZoom());
  }

  prefetchBasemapTiles(center, zoom) {
    if (!this.map || !center || !Number.isFinite(zoom)) return;
    const sourceZoom = this.zoomAnimationState?.zoom ?? this.map.getZoom();
    const targetZoom = Math.max(this.map.getMinZoom(), Math.min(this.map.getMaxZoom(), zoom));
    const tileZoom = Math.max(
      this.map.getMinZoom(),
      Math.min(this.map.getMaxZoom(), targetZoom < sourceZoom ? Math.floor(targetZoom) : Math.ceil(targetZoom))
    );
    const roundedCenter = `${center.lat.toFixed(3)}:${center.lng.toFixed(3)}`;
    const key = `${tileZoom}:${roundedCenter}`;
    if (key === this.lastTilePrefetchKey) return;
    this.lastTilePrefetchKey = key;
    if (this.tilePrefetchFrame) cancelAnimationFrame(this.tilePrefetchFrame);
    this.tilePrefetchFrame = requestAnimationFrame(() => {
      this.tilePrefetchFrame = null;
      prefetchTileRange(this.map, center, tileZoom, BASEMAP_TILE_URL, this.tilePrefetchCache, 18);
    });
  }

  scheduleHeatTilePrewarm(center, zoom = this.map?.getZoom()) {
    if (!this.map || !center || !this.heatLayerVisible()) return;
    const sourceZoom = this.zoomAnimationState?.zoom ?? this.map.getZoom();
    const targetZoom = Math.max(this.map.getMinZoom(), Math.min(this.map.getMaxZoom(), zoom));
    const candidateZooms = [
      targetZoom < sourceZoom ? Math.floor(targetZoom) : Math.ceil(targetZoom),
      Math.floor(sourceZoom) - 1,
      Math.ceil(sourceZoom) + 1,
    ];
    const zooms = [...new Set(candidateZooms)]
      .filter((candidate) => Number.isFinite(candidate))
      .map((candidate) => Math.max(this.map.getMinZoom(), Math.min(this.map.getMaxZoom(), candidate)));
    const prewarm = () => {
      this.heatTilePrewarmIdle = null;
      for (const tileZoom of zooms) {
        prewarmWindHeatTiles(this, center, tileZoom, 6);
      }
    };
    if (this.heatTilePrewarmIdle && window.cancelIdleCallback) window.cancelIdleCallback(this.heatTilePrewarmIdle);
    if (window.requestIdleCallback) {
      this.heatTilePrewarmIdle = window.requestIdleCallback(prewarm, { timeout: 350 });
    } else {
      window.setTimeout(prewarm, 120);
    }
  }

  onZoomEnd() {
    if (this.zoomAnimationState) this.applyZoomTransform(this.map.getCenter(), this.map.getZoom());
    this.zoomAnimating = false;
    this.zoomAnimationState = null;
    this.zoomAnimationMode = null;
    this.shell?.classList.remove("wind-overlay-zooming");
    this.shell?.classList.remove("wind-overlay-zoom-transition");
    if (this.zoomResetFrame) cancelAnimationFrame(this.zoomResetFrame);
    if (this.zoomResetTimer) window.clearTimeout(this.zoomResetTimer);
    if (this.zoomResetIdle && window.cancelIdleCallback) {
      window.cancelIdleCallback(this.zoomResetIdle);
      this.zoomResetIdle = null;
    }
    this.lastParticleDrawTime = null;
    this.lastParticleTime = null;
    this.zoomResetFrame = requestAnimationFrame(() => {
      this.zoomResetFrame = null;
      const finishZoomReset = () => {
        this.zoomResetIdle = null;
        this.zoomResetTimer = null;
        this.shell?.classList.remove("wind-overlay-input-pending");
        this.setOverlayTransform(L.point(0, 0), 1);
        this.reset(null, { preserveParticles: true });
        this.syncCanvasVisibility();
        this.scheduleHeatTilePrewarm(this.map.getCenter(), this.map.getZoom());
      };
      finishZoomReset();
    });
  }

  maybeLoadPriorityCorridors() {
    if (!this.map || this.map.getZoom() < 12) return;
    if (!this.expandedWind) return;
    if (!this.priorityCorridorManifest) {
      if (this.priorityCorridorManifestPromise) return;
      this.priorityCorridorManifestPromise = fetchOptionalJson(PRIORITY_CORRIDORS_URL)
        .then((payload) => {
          this.priorityCorridorManifest = buildPriorityCorridorManifest(payload);
          refreshCoverageStatus(this);
          this.maybeLoadPriorityCorridors();
        })
        .catch(() => null)
        .finally(() => {
          this.priorityCorridorManifestPromise = null;
        });
      return;
    }
    const meta = nearestPriorityCorridorMeta(this.priorityCorridorManifest, this.map.getCenter());
    if (!meta || this.priorityCorridors.corridors.some((corridor) => corridor.id === meta.id) || this.priorityCorridorLoadingIds.has(meta.id)) return;
    this.priorityCorridorLoadingIds.add(meta.id);
    fetchOptionalJson(versionedDataUrl(meta.client_url))
      .then((payload) => {
        const layer = buildPriorityCorridorLayer(payload);
        if (layer?.corridors?.length) {
          const existing = new Set(this.priorityCorridors.corridors.map((corridor) => corridor.id));
          this.priorityCorridors.corridors.push(...layer.corridors.filter((corridor) => !existing.has(corridor.id)));
        }
        this.heatDirty = true;
        this.draw();
        refreshCoverageStatus(this);
      })
      .catch(() => null)
      .finally(() => {
        this.priorityCorridorLoadingIds.delete(meta.id);
      });
  }

  reset(event = null, options = {}) {
    if (this.zoomAnimating && event?.type === "move") {
      if (this.zoomAnimationMode !== "animated") this.applyZoomTransform(this.map.getCenter(), this.map.getZoom());
      return;
    }
    const size = this.map.getSize();
    const metrics = this.overlayRenderMetrics(size);
    this.renderMetrics = metrics;
    const ratio = metrics.pixelRatio;
    // The heat canvas is permanently hidden in the tile-based overlay path. Reallocating its
    // full-screen backing store (canvas.width = …) on every move/pan frame is pure dead work and
    // a real source of mobile pan jank — skip it entirely while hidden.
    if (!this.canvas.hidden) {
      this.canvas.style.left = `${-metrics.pad.x}px`;
      this.canvas.style.top = `${-metrics.pad.y}px`;
      this.canvas.width = Math.ceil(metrics.cssSize.x * ratio);
      this.canvas.height = Math.ceil(metrics.cssSize.y * ratio);
      this.canvas.style.width = `${metrics.cssSize.x}px`;
      this.canvas.style.height = `${metrics.cssSize.y}px`;
      this.ctx.setTransform(ratio, 0, 0, ratio, metrics.pad.x * ratio, metrics.pad.y * ratio);
    }
    this.heatDirty = true;
    if (this.particleCanvas) {
      const particleWidth = Math.ceil(metrics.cssSize.x * ratio);
      const particleHeight = Math.ceil(metrics.cssSize.y * ratio);
      const particleSizeChanged = this.particleCanvas.width !== particleWidth || this.particleCanvas.height !== particleHeight;
      this.particleCanvas.style.left = `${-metrics.pad.x}px`;
      this.particleCanvas.style.top = `${-metrics.pad.y}px`;
      if (particleSizeChanged) {
        this.particleCanvas.width = particleWidth;
        this.particleCanvas.height = particleHeight;
      }
      this.particleCanvas.style.width = `${metrics.cssSize.x}px`;
      this.particleCanvas.style.height = `${metrics.cssSize.y}px`;
      this.particleCtx.setTransform(ratio, 0, 0, ratio, metrics.pad.x * ratio, metrics.pad.y * ratio);
      // While a gesture is active the particle canvas is hidden, so don't reseed it every move
      // frame — onMotionEnd reseeds once the map settles.
      if ((!options.preserveParticles || particleSizeChanged) && !this.motionActive) {
        this.particleCtx.clearRect(-metrics.pad.x, -metrics.pad.y, metrics.cssSize.x, metrics.cssSize.y);
        this.resetParticles();
      }
    }
    if (!this.canvas.hidden) this.draw();
    this.captureRenderView(metrics);
  }

  bounds() {
    if (this.expandedWind?.bounds) {
      return L.latLngBounds([
        [this.expandedWind.bounds.minLat, this.expandedWind.bounds.minLon],
        [this.expandedWind.bounds.maxLat, this.expandedWind.bounds.maxLon],
      ]);
    }
    if (this.localWind?.bounds) {
      return L.latLngBounds([
        [this.localWind.bounds.minLat, this.localWind.bounds.minLon],
        [this.localWind.bounds.maxLat, this.localWind.bounds.maxLon],
      ]);
    }
    return L.latLngBounds(VIEW_BOUNDS);
  }

  gridValue(grid, row, col) {
    const rows = this.step.shape[0];
    const cols = this.step.shape[1];
    const r = Math.max(0, Math.min(rows - 1, row));
    const c = Math.max(0, Math.min(cols - 1, col));
    return grid[r][c];
  }

  bilinear(grid, row, col) {
    const r0 = Math.floor(row);
    const c0 = Math.floor(col);
    const r1 = r0 + 1;
    const c1 = c0 + 1;
    const tr = row - r0;
    const tc = col - c0;
    const v00 = this.gridValue(grid, r0, c0);
    const v10 = this.gridValue(grid, r1, c0);
    const v01 = this.gridValue(grid, r0, c1);
    const v11 = this.gridValue(grid, r1, c1);
    if ([v00, v10, v01, v11].some((value) => value === null || Number.isNaN(value))) return null;
    return (
      v00 * (1 - tr) * (1 - tc) +
      v10 * tr * (1 - tc) +
      v01 * (1 - tr) * tc +
      v11 * tr * tc
    );
  }

  fieldAt(latlng) {
    if (this.visibleLayers.aromepi) return this.aromePiFieldAt(latlng, false);
    if (this.visibleLayers.icon2i) return this.icon2iFieldAt(latlng, false);
    if (this.visibleLayers.moloch) return this.molochFieldAt(latlng, false);
    if (!this.visibleLayers.arome) return null;
    return this.aromeFieldAt(latlng, false);
  }

  rawModelFieldAt(model, step, latlng, contextFallback = false, defaults = {}) {
    if (!model || !step?.shape || !model.bbox_wgs84) return null;
    const [minLon, minLat, maxLon, maxLat] = model.bbox_wgs84;
    if (latlng.lng < minLon || latlng.lng > maxLon || latlng.lat < minLat || latlng.lat > maxLat) return null;
    const rows = step.shape[0];
    const cols = step.shape[1];
    const row = ((maxLat - latlng.lat) / (maxLat - minLat)) * (rows - 1);
    const col = ((latlng.lng - minLon) / (maxLon - minLon)) * (cols - 1);
    const speed = bilinearGrid(step.speed_ms, row, col, rows, cols);
    if (speed === null) return null;
    const meanSpeed = bilinearGrid(step.mean_speed_ms || step.speed_ms, row, col, rows, cols);
    const gustSpeed = bilinearGrid(step.gust_speed_ms, row, col, rows, cols);
    const u = bilinearGrid(step.mean_u_ms || step.u_ms, row, col, rows, cols);
    const v = bilinearGrid(step.mean_v_ms || step.v_ms, row, col, rows, cols);
    const flowToDeg = u === null || v === null ? null : windDirectionToDeg(u, v);
    const windFromDeg = flowToDeg === null ? null : (flowToDeg + 180) % 360;
    const particleSpeed = meanSpeed ?? speed;
    return {
      speed,
      speedKnots: speed * KNOTS_PER_MPS,
      meanSpeed,
      meanSpeedKnots: meanSpeed === null ? null : meanSpeed * KNOTS_PER_MPS,
      gustSpeed,
      gustSpeedKnots: gustSpeed === null ? null : gustSpeed * KNOTS_PER_MPS,
      particleSpeed,
      particleSpeedKnots: particleSpeed * KNOTS_PER_MPS,
      baseSpeed: speed,
      confidence: defaults.confidence ?? 0.42,
      modelConfidence: defaults.confidence ?? 0.42,
      renderAlpha: contextFallback ? 0.42 : defaults.renderAlpha ?? 0.62,
      sourceType: defaults.sourceType || "raw",
      sourceLabel: defaults.sourceLabel || model.model_label || "Modèle brut",
      heightLabel: defaults.heightLabel || `${model.height_agl_m || 10} m AGL`,
      resolutionLabel: defaults.resolutionLabel || model.resolution || "~1 km",
      windFromDeg,
      flowToDeg,
      contextFallback,
    };
  }

  aromeFieldAt(latlng, contextFallback = false) {
    return this.rawModelFieldAt(
      this.arome,
      this.aromeStep,
      latlng,
      contextFallback,
      { sourceType: "arome", sourceLabel: "AROME contexte Corse", heightLabel: "10 m AGL", resolutionLabel: "~1 km" }
    );
  }

  molochFieldAt(latlng, contextFallback = false) {
    return this.rawModelFieldAt(this.moloch, this.molochStep, latlng, contextFallback, {
      sourceType: "moloch",
      sourceLabel: "MOLOCH Italie",
      heightLabel: "10 m AGL",
      resolutionLabel: "1.2 km",
      confidence: 0.4,
      renderAlpha: 0.58,
    });
  }

  icon2iFieldAt(latlng, contextFallback = false) {
    return this.rawModelFieldAt(this.icon2i, this.icon2iStep, latlng, contextFallback, {
      sourceType: "icon2i",
      sourceLabel: "ICON-2I Italie",
      heightLabel: "10 m AGL",
      resolutionLabel: "2.2 km",
      confidence: 0.4,
      renderAlpha: 0.58,
    });
  }

  aromePiFieldAt(latlng, contextFallback = false) {
    return this.rawModelFieldAt(this.aromepi, this.aromePiStep, latlng, contextFallback, {
      sourceType: "aromepi",
      sourceLabel: "AROME-PI immediat",
      heightLabel: "10 m AGL",
      resolutionLabel: "vent moyen 2.5 km / rafales 1 km",
      confidence: 0.48,
      renderAlpha: 0.66,
    });
  }

  localFieldAt(latlng) {
    if (!this.localWind || !this.localStep) return null;
    const { minLon, minLat, maxLon, maxLat } = this.localWind.bounds;
    if (latlng.lng < minLon || latlng.lng > maxLon || latlng.lat < minLat || latlng.lat > maxLat) return null;
    const rows = this.localStep.shape[0];
    const cols = this.localStep.shape[1];
    const row = ((maxLat - latlng.lat) / (maxLat - minLat)) * (rows - 1);
    const col = ((latlng.lng - minLon) / (maxLon - minLon)) * (cols - 1);
    const edge = Math.min(row / (rows - 1), col / (cols - 1), (rows - 1 - row) / (rows - 1), (cols - 1 - col) / (cols - 1));
    const speedKnots = bilinearGrid(this.localStep.speed_kt, row, col, rows, cols);
    if (speedKnots === null) return null;
    const staticFields = this.localWind.static_fields || {};
    const surfaceClassId = staticFields.surface_class
      ? nearestGridValue(staticFields.surface_class, row, col, rows, cols)
      : 0;
    const surfaceConfidence = staticFields.static_confidence
      ? bilinearGrid(staticFields.static_confidence, row, col, rows, cols) || 0
      : 0;
    const distanceToCoastM = staticFields.distance_to_coast_m
      ? bilinearGrid(staticFields.distance_to_coast_m, row, col, rows, cols) || 0
      : null;
    const coastalBand = staticFields.coastal_band_index
      ? bilinearGrid(staticFields.coastal_band_index, row, col, rows, cols) || 0
      : 0;
    const sourceRenderAlpha = this.displayMode === "surface" ? 0.9 : bilinearGrid(this.localStep.render_alpha, row, col, rows, cols) ?? 1;
    const domainFeather = smoothstep(0.1, 0.3, edge);
    if (sourceRenderAlpha * domainFeather < 0.055) return null;
    return {
      speed: speedKnots / KNOTS_PER_MPS,
      speedKnots,
      baseSpeed: speedKnots / KNOTS_PER_MPS,
      devente: bilinearGrid(this.localStep.devente_index, row, col, rows, cols) || 0,
      acceleration: bilinearGrid(this.localStep.acceleration_index, row, col, rows, cols) || 0,
      turbulence: bilinearGrid(this.localStep.turbulence_proxy, row, col, rows, cols) || 0,
      confidence: applyValidationConfidence(bilinearGrid(this.localStep.confidence, row, col, rows, cols) || 0, this.validation),
      modelConfidence: bilinearGrid(this.localStep.confidence, row, col, rows, cols) || 0,
      quality: applyValidationQuality(bilinearGrid(this.localStep.windsurf_quality_index, row, col, rows, cols) || 0, this.validation),
      modelQuality: bilinearGrid(this.localStep.windsurf_quality_index, row, col, rows, cols) || 0,
      sessionClassId: nearestGridValue(this.localStep.session_class_id, row, col, rows, cols),
      ratio: bilinearGrid(this.localStep.ratio_vs_arome_10m, row, col, rows, cols) || 1,
      renderAlpha: sourceRenderAlpha,
      sourceFidelity: Math.round(bilinearGrid(this.localStep.source_fidelity, row, col, rows, cols) || 0),
      surfaceClassId,
      surfaceConfidence,
      distanceToCoastM,
      coastalBand,
      domainFeather,
      sourceType: "local",
      sourceLabel: "Downscale Ajaccio",
      heightLabel: "2 m",
      resolutionLabel: "60 m",
      local2m: true,
    };
  }

  expandedFieldAt(latlng) {
    if (!this.expandedWind || !this.expandedStep) return null;
    const { minLon, minLat, maxLon, maxLat } = this.expandedWind.bounds;
    if (latlng.lng < minLon || latlng.lng > maxLon || latlng.lat < minLat || latlng.lat > maxLat) return null;
    const rows = this.expandedStep.shape[0];
    const cols = this.expandedStep.shape[1];
    const row = ((maxLat - latlng.lat) / (maxLat - minLat)) * (rows - 1);
    const col = ((latlng.lng - minLon) / (maxLon - minLon)) * (cols - 1);
    const edge = Math.min(row / (rows - 1), col / (cols - 1), (rows - 1 - row) / (rows - 1), (cols - 1 - col) / (cols - 1));
    const speedKnots = bilinearGrid(this.expandedStep.speed_kt, row, col, rows, cols);
    if (speedKnots === null) return null;
    const staticFields = this.expandedWind.static_fields || {};
    const surfaceClassId = staticFields.surface_class
      ? nearestGridValue(staticFields.surface_class, row, col, rows, cols)
      : 0;
    const surfaceConfidence = staticFields.static_confidence
      ? bilinearGrid(staticFields.static_confidence, row, col, rows, cols) || 0
      : 0;
    const distanceToCoastM = staticFields.distance_to_coast_m
      ? bilinearGrid(staticFields.distance_to_coast_m, row, col, rows, cols) || 0
      : null;
    const coastalBand = staticFields.coastal_band_index
      ? bilinearGrid(staticFields.coastal_band_index, row, col, rows, cols) || 0
      : 0;
    const renderAlpha = staticFields.render_alpha
      ? bilinearGrid(staticFields.render_alpha, row, col, rows, cols) ?? 0
      : 0.44;
    const domainFeather = smoothstep(0.08, 0.26, edge);
    if (renderAlpha * domainFeather < 0.055) return null;
    const u = bilinearGrid(this.expandedStep.u_ms, row, col, rows, cols);
    const v = bilinearGrid(this.expandedStep.v_ms, row, col, rows, cols);
    const flowToDeg = u === null || v === null ? null : windDirectionToDeg(u, v);
    const windFromDeg = flowToDeg === null ? null : (flowToDeg + 180) % 360;
    return {
      speed: speedKnots / KNOTS_PER_MPS,
      speedKnots,
      baseSpeed: speedKnots / KNOTS_PER_MPS,
      devente: bilinearGrid(this.expandedStep.devente_index, row, col, rows, cols) || 0,
      acceleration: bilinearGrid(this.expandedStep.acceleration_index, row, col, rows, cols) || 0,
      turbulence: bilinearGrid(this.expandedStep.turbulence_proxy, row, col, rows, cols) || 0,
      confidence: applyValidationConfidence(bilinearGrid(this.expandedStep.confidence, row, col, rows, cols) || 0, this.validation),
      modelConfidence: bilinearGrid(this.expandedStep.confidence, row, col, rows, cols) || 0,
      quality: applyValidationQuality(bilinearGrid(this.expandedStep.windsurf_quality_index, row, col, rows, cols) || 0, this.validation),
      modelQuality: bilinearGrid(this.expandedStep.windsurf_quality_index, row, col, rows, cols) || 0,
      sessionClassId: nearestGridValue(this.expandedStep.session_class_id, row, col, rows, cols),
      ratio: bilinearGrid(this.expandedStep.ratio_vs_arome_10m, row, col, rows, cols) || 1,
      renderAlpha,
      sourceFidelity: Math.round(bilinearGrid(this.expandedStep.source_fidelity, row, col, rows, cols) || 0),
      surfaceClassId,
      surfaceConfidence,
      distanceToCoastM,
      coastalBand,
      domainFeather,
      sourceType: "expanded",
      sourceLabel: "Baie Ajaccio 5 km",
      heightLabel: "2 m cible",
      resolutionLabel: "120 m",
      local2m: true,
      overview2m: true,
      windFromDeg,
      flowToDeg,
    };
  }

  spotFieldAt(latlng) {
    if (!this.spotGrids?.spots?.length || !this.localStep || this.map?.getZoom() < 13) return null;
    for (const spot of this.spotGrids.spots) {
      if (latlng.lng < spot.bounds.minLon || latlng.lng > spot.bounds.maxLon || latlng.lat < spot.bounds.minLat || latlng.lat > spot.bounds.maxLat) {
        continue;
      }
      const step = spot.forecast_steps?.[this.stepIndex];
      if (!step) continue;
      const rows = step.shape[0];
      const cols = step.shape[1];
      const row = ((spot.bounds.maxLat - latlng.lat) / (spot.bounds.maxLat - spot.bounds.minLat)) * (rows - 1);
      const col = ((latlng.lng - spot.bounds.minLon) / (spot.bounds.maxLon - spot.bounds.minLon)) * (cols - 1);
      const edge = Math.min(row / (rows - 1), col / (cols - 1), (rows - 1 - row) / (rows - 1), (cols - 1 - col) / (cols - 1));
      const speedKnots = bilinearGrid(step.speed_kt, row, col, rows, cols);
      if (speedKnots === null) continue;
      const renderAlpha = bilinearGrid(step.render_alpha, row, col, rows, cols) ?? 1;
      const domainFeather = smoothstep(0.08, 0.24, edge);
      if (renderAlpha * domainFeather < 0.055) continue;
      return {
        speed: speedKnots / KNOTS_PER_MPS,
        speedKnots,
        baseSpeed: speedKnots / KNOTS_PER_MPS,
        devente: bilinearGrid(step.devente_index, row, col, rows, cols) || 0,
        acceleration: bilinearGrid(step.acceleration_index, row, col, rows, cols) || 0,
        turbulence: bilinearGrid(step.turbulence_proxy, row, col, rows, cols) || 0,
        confidence: applyValidationConfidence(bilinearGrid(step.confidence, row, col, rows, cols) || 0, this.validation),
        modelConfidence: bilinearGrid(step.confidence, row, col, rows, cols) || 0,
        quality: applyValidationQuality(bilinearGrid(step.windsurf_quality_index, row, col, rows, cols) || 0, this.validation),
        modelQuality: bilinearGrid(step.windsurf_quality_index, row, col, rows, cols) || 0,
        sessionClassId: nearestGridValue(step.session_class_id, row, col, rows, cols),
        ratio: bilinearGrid(step.ratio_vs_arome_10m, row, col, rows, cols) || 1,
        renderAlpha,
        sourceFidelity: 2,
        domainFeather,
        sourceType: "spot",
        sourceLabel: "Spot nested",
        heightLabel: "2 m",
        resolutionLabel: `${spot.resolution_m} m`,
        local2m: true,
        spot: true,
        spotLabel: spot.label,
        resolutionM: spot.resolution_m,
      };
    }
    return null;
  }

  windNinjaSpotFieldAt(latlng) {
    if (!this.windNinjaSpots?.spots?.length || this.map?.getZoom() < 13) return null;
    for (const spot of this.windNinjaSpots.spots) {
      if (latlng.lng < spot.bounds.minLon || latlng.lng > spot.bounds.maxLon || latlng.lat < spot.bounds.minLat || latlng.lat > spot.bounds.maxLat) {
        continue;
      }
      const step = spot.forecast_steps?.find((candidate) => Number(candidate.lead_hour) === Number(this.step?.lead_hour)) || spot.forecast_steps?.[0];
      if (!step) continue;
      const rows = step.shape[0];
      const cols = step.shape[1];
      const row = ((spot.bounds.maxLat - latlng.lat) / (spot.bounds.maxLat - spot.bounds.minLat)) * (rows - 1);
      const col = ((latlng.lng - spot.bounds.minLon) / (spot.bounds.maxLon - spot.bounds.minLon)) * (cols - 1);
      const edge = Math.min(row / (rows - 1), col / (cols - 1), (rows - 1 - row) / (rows - 1), (cols - 1 - col) / (cols - 1));
      const speedKnots = bilinearGrid(step.speed_kt, row, col, rows, cols);
      if (speedKnots === null) continue;
      const renderAlpha = bilinearGrid(step.render_alpha, row, col, rows, cols) ?? 1;
      const domainFeather = smoothstep(0.06, 0.2, edge);
      if (renderAlpha * domainFeather < 0.055) continue;
      return {
        speed: speedKnots / KNOTS_PER_MPS,
        speedKnots,
        baseSpeed: speedKnots / KNOTS_PER_MPS,
        devente: bilinearGrid(step.devente_index, row, col, rows, cols) || 0,
        acceleration: bilinearGrid(step.acceleration_index, row, col, rows, cols) || 0,
        turbulence: bilinearGrid(step.turbulence_proxy, row, col, rows, cols) || 0,
        confidence: applyValidationConfidence(bilinearGrid(step.confidence, row, col, rows, cols) || 0, this.validation),
        modelConfidence: bilinearGrid(step.confidence, row, col, rows, cols) || 0,
        quality: applyValidationQuality(bilinearGrid(step.windsurf_quality_index, row, col, rows, cols) || 0, this.validation),
        modelQuality: bilinearGrid(step.windsurf_quality_index, row, col, rows, cols) || 0,
        sessionClassId: nearestGridValue(step.session_class_id, row, col, rows, cols),
        ratio: bilinearGrid(step.ratio_vs_parent, row, col, rows, cols) || 1,
        renderAlpha,
        sourceFidelity: 4,
        domainFeather,
        sourceType: "windninja",
        sourceLabel: "WindNinja 5 m",
        heightLabel: `${spot.height_m || 2} m`,
        resolutionLabel: `${spot.resolution_m} m`,
        local2m: true,
        windNinja: true,
        spot: true,
        spotLabel: spot.label,
        resolutionM: spot.resolution_m,
      };
    }
    return null;
  }

  corridorFieldAt(latlng) {
    if (!this.priorityCorridors?.corridors?.length || this.map?.getZoom() < 12) return null;
    for (const corridor of this.priorityCorridors.corridors) {
      if (
        latlng.lng < corridor.bounds.minLon ||
        latlng.lng > corridor.bounds.maxLon ||
        latlng.lat < corridor.bounds.minLat ||
        latlng.lat > corridor.bounds.maxLat
      ) {
        continue;
      }
      const step = corridor.forecast_steps?.[this.stepIndex];
      if (!step) continue;
      const rows = step.shape[0];
      const cols = step.shape[1];
      const row = ((corridor.bounds.maxLat - latlng.lat) / (corridor.bounds.maxLat - corridor.bounds.minLat)) * (rows - 1);
      const col = ((latlng.lng - corridor.bounds.minLon) / (corridor.bounds.maxLon - corridor.bounds.minLon)) * (cols - 1);
      const edge = Math.min(row / (rows - 1), col / (cols - 1), (rows - 1 - row) / (rows - 1), (cols - 1 - col) / (cols - 1));
      const speedKnots = bilinearGrid(step.speed_kt, row, col, rows, cols);
      if (speedKnots === null) continue;
      const renderAlpha = bilinearGrid(step.render_alpha, row, col, rows, cols) ?? 0;
      const domainFeather = smoothstep(0.08, 0.24, edge);
      if (renderAlpha * domainFeather < 0.055) continue;
      const staticFields = corridor.static_fields || {};
      const surfaceClassId = staticFields.surface_class
        ? nearestGridValue(staticFields.surface_class, row, col, rows, cols)
        : 0;
      const surfaceConfidence = staticFields.static_confidence
        ? bilinearGrid(staticFields.static_confidence, row, col, rows, cols) || 0
        : 0;
      const distanceToCoastM = staticFields.distance_to_coast_m
        ? bilinearGrid(staticFields.distance_to_coast_m, row, col, rows, cols) || 0
        : null;
      const coastalBand = staticFields.coastal_band_index
        ? bilinearGrid(staticFields.coastal_band_index, row, col, rows, cols) || 0
        : 0;
      const u = bilinearGrid(step.u_ms, row, col, rows, cols);
      const v = bilinearGrid(step.v_ms, row, col, rows, cols);
      const flowToDeg = u === null || v === null ? null : windDirectionToDeg(u, v);
      const windFromDeg = flowToDeg === null ? null : (flowToDeg + 180) % 360;
      return {
        speed: speedKnots / KNOTS_PER_MPS,
        speedKnots,
        baseSpeed: speedKnots / KNOTS_PER_MPS,
        devente: bilinearGrid(step.devente_index, row, col, rows, cols) || 0,
        acceleration: bilinearGrid(step.acceleration_index, row, col, rows, cols) || 0,
        turbulence: bilinearGrid(step.turbulence_proxy, row, col, rows, cols) || 0,
        confidence: applyValidationConfidence(bilinearGrid(step.confidence, row, col, rows, cols) || 0, this.validation),
        modelConfidence: bilinearGrid(step.confidence, row, col, rows, cols) || 0,
        quality: applyValidationQuality(bilinearGrid(step.windsurf_quality_index, row, col, rows, cols) || 0, this.validation),
        modelQuality: bilinearGrid(step.windsurf_quality_index, row, col, rows, cols) || 0,
        sessionClassId: nearestGridValue(step.session_class_id, row, col, rows, cols),
        ratio: bilinearGrid(step.ratio_vs_parent, row, col, rows, cols) || 1,
        renderAlpha,
        sourceFidelity: Math.round(bilinearGrid(step.source_fidelity, row, col, rows, cols) || 1),
        domainFeather,
        sourceType: "corridor",
        sourceLabel: "Corridor prioritaire",
        heightLabel: "2 m cible",
        resolutionLabel: `${corridor.resolution_m} m`,
        local2m: true,
        corridor: true,
        corridorLabel: corridor.label,
        surfaceClassId,
        surfaceConfidence,
        distanceToCoastM,
        coastalBand,
        windFromDeg,
        flowToDeg,
      };
    }
    return null;
  }

  cfdCorrectionAt(latlng) {
    if (!this.cfd) return null;
    const latPad = CFD_INFLUENCE_RADIUS_M / 111_320;
    const lonPad = latPad / Math.max(0.25, Math.cos(degToRad(latlng.lat)));
    if (
      latlng.lat < this.cfd.bounds.minLat - latPad ||
      latlng.lat > this.cfd.bounds.maxLat + latPad ||
      latlng.lng < this.cfd.bounds.minLon - lonPad ||
      latlng.lng > this.cfd.bounds.maxLon + lonPad
    ) {
      return null;
    }

    let weightSum = 0;
    let ratioSum = 0;
    let nearest = Infinity;
    for (const sample of this.cfd.samples) {
      const distance = haversineMeters(latlng.lat, latlng.lng, sample.lat, sample.lng);
      if (distance > CFD_INFLUENCE_RADIUS_M) continue;
      nearest = Math.min(nearest, distance);
      const weight = Math.exp(-(distance * distance) / (2 * CFD_SIGMA_M * CFD_SIGMA_M));
      weightSum += weight;
      ratioSum += sample.ratio * weight;
    }

    if (weightSum <= 0 || !Number.isFinite(nearest)) return null;
    const weightedRatio = ratioSum / weightSum;
    const confidence = 1 - smoothstep(45, CFD_INFLUENCE_RADIUS_M, nearest);
    return {
      ratio: 1 + (weightedRatio - 1) * Math.max(0.18, confidence),
      confidence,
      nearestM: nearest,
    };
  }

  drawHeat() {
    // The visible colour overlay is rendered by the Leaflet GridLayer (heatTileLayer).
    // This fullscreen canvas is kept permanently hidden (see syncCanvasVisibility), so the
    // expensive per-pixel raster below would only ever paint into an invisible buffer.
    // Skip it while hidden to avoid recomputing the whole field on every pan/move.
    if (this.canvas?.hidden) return;
    if (!anyRawLayerVisible(this)) return;
    if (this.displayMode !== "speed") return;
    const size = this.map.getSize();
    const metrics = this.renderMetrics || this.overlayRenderMetrics(size);
    if (this.heatDirty) {
      this.drawHeatRaster(size, metrics);
      this.heatDirty = false;
    }
    this.ctx.save();
    this.ctx.globalCompositeOperation = "source-over";
    this.ctx.filter =
      this.displayMode === "quality"
        ? "blur(1.5px) saturate(1.12) contrast(1.04)"
        : "blur(6px) saturate(1.45) contrast(1.18)";
    this.ctx.globalAlpha = 1;
    this.ctx.drawImage(this.heatCanvas, -metrics.pad.x, -metrics.pad.y, metrics.cssSize.x, metrics.cssSize.y);
    this.ctx.restore();
  }

  drawMultiscalePlan() {
    if (!this.multiscalePlan?.targetBounds) return;
    const zoom = this.map.getZoom();
    if (zoom < 8) return;
    this.ctx.save();
    this.drawBoundsFootprint(
      this.multiscalePlan.targetBounds,
      "rgba(250, 204, 21, 0.055)",
      "rgba(250, 204, 21, 0.86)",
      zoom >= 11 ? 2.4 : 1.8,
      [12, 7],
    );
    if (this.multiscalePlan.currentBounds) {
      this.drawBoundsFootprint(
        this.multiscalePlan.currentBounds,
        "rgba(103, 232, 249, 0.035)",
        "rgba(103, 232, 249, 0.66)",
        zoom >= 11 ? 1.7 : 1.2,
        [5, 5],
      );
    }
    this.ctx.restore();
  }

  drawBoundsFootprint(bounds, fillStyle, strokeStyle, lineWidth, dash) {
    const corners = [
      [bounds.minLat, bounds.minLon],
      [bounds.minLat, bounds.maxLon],
      [bounds.maxLat, bounds.maxLon],
      [bounds.maxLat, bounds.minLon],
    ].map((point) => this.map.latLngToContainerPoint(point));
    if (!this.pointsNearViewport(corners)) return;
    this.ctx.beginPath();
    corners.forEach((point, index) => {
      if (index === 0) this.ctx.moveTo(point.x, point.y);
      else this.ctx.lineTo(point.x, point.y);
    });
    this.ctx.closePath();
    this.ctx.fillStyle = fillStyle;
    this.ctx.strokeStyle = strokeStyle;
    this.ctx.lineWidth = lineWidth;
    this.ctx.setLineDash(dash);
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.setLineDash([]);
  }

  drawSpotFootprints() {
    if (!this.spotGrids?.spots?.length) return;
    if (this.displayMode !== "confidence") return;
    const zoom = this.map.getZoom();
    if (zoom < 11) return;
    this.ctx.save();
    this.ctx.lineWidth = zoom >= 13 ? 1.8 : 1.2;
    this.ctx.setLineDash(zoom >= 13 ? [] : [7, 5]);
    for (const spot of this.spotGrids.spots) {
      const corners = [
        [spot.bounds.minLat, spot.bounds.minLon],
        [spot.bounds.minLat, spot.bounds.maxLon],
        [spot.bounds.maxLat, spot.bounds.maxLon],
        [spot.bounds.maxLat, spot.bounds.minLon],
      ].map((point) => this.map.latLngToContainerPoint(point));
      if (!this.pointsNearViewport(corners)) continue;
      this.ctx.beginPath();
      corners.forEach((point, index) => {
        if (index === 0) this.ctx.moveTo(point.x, point.y);
        else this.ctx.lineTo(point.x, point.y);
      });
      this.ctx.closePath();
      const isFine = spot.resolution_m <= 10;
      this.ctx.fillStyle = isFine ? "rgba(103, 232, 249, 0.055)" : "rgba(245, 202, 66, 0.045)";
      this.ctx.strokeStyle = isFine ? "rgba(103, 232, 249, 0.72)" : "rgba(245, 202, 66, 0.58)";
      this.ctx.fill();
      this.ctx.stroke();
    }
    this.ctx.restore();
  }

  drawPriorityCorridorFootprints() {
    if (!this.priorityCorridors?.corridors?.length) return;
    if (this.displayMode !== "confidence") return;
    const zoom = this.map.getZoom();
    if (zoom < 12) return;
    this.ctx.save();
    this.ctx.lineWidth = zoom >= 13 ? 2.0 : 1.35;
    this.ctx.setLineDash([8, 5]);
    for (const corridor of this.priorityCorridors.corridors) {
      const corners = [
        [corridor.bounds.minLat, corridor.bounds.minLon],
        [corridor.bounds.minLat, corridor.bounds.maxLon],
        [corridor.bounds.maxLat, corridor.bounds.maxLon],
        [corridor.bounds.maxLat, corridor.bounds.minLon],
      ].map((point) => this.map.latLngToContainerPoint(point));
      if (!this.pointsNearViewport(corners)) continue;
      this.ctx.beginPath();
      corners.forEach((point, index) => {
        if (index === 0) this.ctx.moveTo(point.x, point.y);
        else this.ctx.lineTo(point.x, point.y);
      });
      this.ctx.closePath();
      this.ctx.fillStyle = "rgba(251, 191, 36, 0.055)";
      this.ctx.strokeStyle = "rgba(251, 191, 36, 0.82)";
      this.ctx.fill();
      this.ctx.stroke();
    }
    this.ctx.restore();
  }

  drawBayModelPlan() {
    if (!this.bayModel?.tiles?.length) return;
    const zoom = this.map.getZoom();
    if (zoom < 9) return;
    this.ctx.save();
    this.drawBayCorridor(zoom);
    const visibleLimit = zoom >= 13 ? 120 : zoom >= 11 ? 72 : 36;
    const tiles = this.bayModel.tiles.slice(0, visibleLimit);
    this.ctx.lineJoin = "round";
    for (const tile of tiles) {
      const points = tile.polygon.map((point) => this.map.latLngToContainerPoint([point.lat, point.lng]));
      if (points.length < 3 || !this.pointsNearViewport(points)) continue;
      const style = bayTierStyle(tile.fidelityId);
      const rankIntensity = 1 - Math.min(1, (tile.rank - 1) / 120);
      this.ctx.beginPath();
      points.forEach((point, index) => {
        if (index === 0) this.ctx.moveTo(point.x, point.y);
        else this.ctx.lineTo(point.x, point.y);
      });
      this.ctx.closePath();
      this.ctx.fillStyle = `rgba(${style.rgb}, ${style.fill + rankIntensity * 0.08})`;
      this.ctx.strokeStyle = `rgba(${style.rgb}, ${style.stroke})`;
      this.ctx.lineWidth = style.width + (zoom >= 13 ? 0.55 : 0);
      this.ctx.fill();
      this.ctx.stroke();
    }
    this.ctx.restore();
  }

  drawBayCorridor(zoom) {
    const corridor = this.bayModel.corridor.map((point) => this.map.latLngToContainerPoint([point.lat, point.lng]));
    if (corridor.length < 2 || !this.pointsNearViewport(corridor)) return;
    this.ctx.save();
    this.ctx.beginPath();
    corridor.forEach((point, index) => {
      if (index === 0) this.ctx.moveTo(point.x, point.y);
      else this.ctx.lineTo(point.x, point.y);
    });
    this.ctx.strokeStyle = "rgba(248, 250, 252, 0.76)";
    this.ctx.lineWidth = zoom >= 12 ? 2.2 : 1.5;
    this.ctx.setLineDash([10, 7]);
    this.ctx.stroke();
    for (const point of corridor) {
      this.ctx.beginPath();
      this.ctx.arc(point.x, point.y, zoom >= 12 ? 4.2 : 3.2, 0, Math.PI * 2);
      this.ctx.fillStyle = "rgba(103, 232, 249, 0.92)";
      this.ctx.fill();
      this.ctx.strokeStyle = "rgba(6, 17, 31, 0.7)";
      this.ctx.lineWidth = 1.2;
      this.ctx.stroke();
    }
    this.ctx.restore();
  }

  drawCoastalTiles() {
    if (!this.coastalTiles?.tiles?.length) return;
    const zoom = this.map.getZoom();
    if (zoom < 10) return;
    const visibleLimit = zoom >= 13 ? 120 : 48;
    const tiles = this.coastalTiles.tiles.slice(0, visibleLimit);
    this.ctx.save();
    this.ctx.lineJoin = "round";
    for (const tile of tiles) {
      const points = tile.polygon.map((point) => this.map.latLngToContainerPoint([point.lat, point.lng]));
      if (points.length < 3 || !this.pointsNearViewport(points)) continue;
      const priority = Math.max(0.18, Math.min(1, tile.priority * 1.9));
      this.ctx.beginPath();
      points.forEach((point, index) => {
        if (index === 0) this.ctx.moveTo(point.x, point.y);
        else this.ctx.lineTo(point.x, point.y);
      });
      this.ctx.closePath();
      this.ctx.fillStyle = `rgba(167, 139, 250, ${0.12 + priority * 0.12})`;
      this.ctx.strokeStyle = `rgba(245, 243, 255, ${0.38 + priority * 0.5})`;
      this.ctx.lineWidth = zoom >= 13 ? 1.7 : 1.25;
      this.ctx.fill();
      this.ctx.stroke();
    }
    this.ctx.restore();
  }

  drawSolvedCfdFootprint() {
    if (!this.cfd) return;
    const zoom = this.map.getZoom();
    if (zoom < 12) return;
    const padMeters = zoom >= 15 ? 45 : 85;
    const centerLat = (this.cfd.bounds.minLat + this.cfd.bounds.maxLat) / 2;
    const latPad = padMeters / 111_320;
    const lonPad = latPad / Math.max(0.25, Math.cos(degToRad(centerLat)));
    const corners = [
      [this.cfd.bounds.minLat - latPad, this.cfd.bounds.minLon - lonPad],
      [this.cfd.bounds.minLat - latPad, this.cfd.bounds.maxLon + lonPad],
      [this.cfd.bounds.maxLat + latPad, this.cfd.bounds.maxLon + lonPad],
      [this.cfd.bounds.maxLat + latPad, this.cfd.bounds.minLon - lonPad],
    ].map((point) => this.map.latLngToContainerPoint(point));
    if (!this.pointsNearViewport(corners)) return;
    this.ctx.save();
    this.ctx.beginPath();
    corners.forEach((point, index) => {
      if (index === 0) this.ctx.moveTo(point.x, point.y);
      else this.ctx.lineTo(point.x, point.y);
    });
    this.ctx.closePath();
    this.ctx.fillStyle = "rgba(217, 70, 239, 0.13)";
    this.ctx.strokeStyle = "rgba(250, 232, 255, 0.86)";
    this.ctx.lineWidth = zoom >= 15 ? 2.4 : 1.8;
    this.ctx.setLineDash([7, 5]);
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.restore();
  }

  pointsNearViewport(points) {
    const size = this.map.getSize();
    const pad = 80;
    return points.some((point) => point.x >= -pad && point.x <= size.x + pad && point.y >= -pad && point.y <= size.y + pad);
  }

  drawHeatRaster(size, metrics = this.overlayRenderMetrics(size)) {
    const scale = size.x < 700 ? 0.24 : 0.28;
    const width = Math.max(280, Math.round(metrics.cssSize.x * scale));
    const height = Math.max(190, Math.round(metrics.cssSize.y * scale));
    this.heatCanvas.width = width;
    this.heatCanvas.height = height;
    const image = this.heatCtx.createImageData(width, height);
    const fields = new Array(width * height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const px = (x / width) * metrics.cssSize.x - metrics.pad.x;
        const py = (y / height) * metrics.cssSize.y - metrics.pad.y;
        const field = this.fieldAt(this.map.containerPointToLatLng([px, py]));
        const index = y * width + x;
        fields[index] = field;
      }
    }

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const field = fields[index];
        const offset = index * 4;
        if (!field) {
          image.data[offset + 3] = 0;
          continue;
        }
        const render = renderFieldColor(field, this.displayMode, this.scaleMaxKnots);
        const rgb = render.rgb;
        const renderAlpha = Math.max(0, Math.min(1, field.renderAlpha ?? 1));
        if (renderAlpha < 0.08) {
          image.data[offset + 3] = 0;
          continue;
        }
        image.data[offset] = rgb[0];
        image.data[offset + 1] = rgb[1];
        image.data[offset + 2] = rgb[2];
        const corridorAlpha = render.alpha;
        image.data[offset + 3] = Math.round(
            Math.min(236, corridorAlpha) *
            (field.domainFeather ?? 1) *
            renderAlpha
        );
      }
    }
    this.heatCtx.putImageData(image, 0, 0);
  }

  drawWindNinjaAnomalyOverlay(size) {
    if (!this.windNinjaSpots?.spots?.length || this.map.getZoom() < 13 || this.displayMode === "surface") return;
    const scale = size.x < 700 ? 0.34 : 0.42;
    const width = Math.max(280, Math.round(size.x * scale));
    const height = Math.max(190, Math.round(size.y * scale));
    this.heatCanvas.width = width;
    this.heatCanvas.height = height;
    const image = this.heatCtx.createImageData(width, height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const px = (x / width) * size.x;
        const py = (y / height) * size.y;
        const field = this.windNinjaSpotFieldAt(this.map.containerPointToLatLng([px, py]));
        const offset = (y * width + x) * 4;
        if (!field) {
          image.data[offset + 3] = 0;
          continue;
        }
        const render = renderFieldColor(field, this.displayMode, this.scaleMaxKnots);
        const alpha = Math.max(0, Math.min(1, field.renderAlpha ?? 0)) * Math.max(0, Math.min(1, field.domainFeather ?? 1));
        if (alpha < 0.08) {
          image.data[offset + 3] = 0;
          continue;
        }
        image.data[offset] = render.rgb[0];
        image.data[offset + 1] = render.rgb[1];
        image.data[offset + 2] = render.rgb[2];
        image.data[offset + 3] = Math.round(Math.min(210, render.alpha) * alpha);
      }
    }
    this.heatCtx.putImageData(image, 0, 0);
    this.ctx.save();
    this.ctx.filter = "blur(2.8px) saturate(1.3) contrast(1.08)";
    this.ctx.drawImage(this.heatCanvas, 0, 0, size.x, size.y);
    this.ctx.restore();
  }

  draw() {
    if (!this.canvas) return;
    const size = this.map.getSize();
    const metrics = this.renderMetrics || this.overlayRenderMetrics(size);
    this.ctx.clearRect(-metrics.pad.x, -metrics.pad.y, metrics.cssSize.x, metrics.cssSize.y);
    if (anyRawLayerVisible(this) && this.displayMode === "speed") this.drawHeat();
    if (!this.zoomAnimating) this.captureRenderView(metrics);
  }

  startParticleLoop() {
    if (this.particleFrame) cancelAnimationFrame(this.particleFrame);
    const tick = (timestamp) => {
      this.animateParticles(timestamp);
      this.particleFrame = requestAnimationFrame(tick);
    };
    this.particleFrame = requestAnimationFrame(tick);
  }

  resetParticles() {
    this.particles = [];
    this.lastParticleDrawTime = null;
    this.lastParticleTime = null;
    if (this.particleCtx && this.map) {
      const size = this.map.getSize();
      const metrics = this.renderMetrics || this.overlayRenderMetrics(size);
      this.particleCtx.clearRect(-metrics.pad.x, -metrics.pad.y, metrics.cssSize.x, metrics.cssSize.y);
    }
  }

  particleFieldAt(latlng) {
    const field = this.visibleLayers.aromepi
      ? this.aromePiFieldAt(latlng, true)
      : this.visibleLayers.icon2i
      ? this.icon2iFieldAt(latlng, true)
      : this.visibleLayers.moloch
        ? this.molochFieldAt(latlng, true)
        : this.aromeFieldAt(latlng, true);
    if (!field || field.flowToDeg === null || field.flowToDeg === undefined) return null;
    const windNinjaSample = this.windNinjaDataSampleAt(latlng);
    if (windNinjaSample) {
      field.speedKnots = windNinjaSample.speedKt;
      field.speed = windNinjaSample.speedKt / KNOTS_PER_MPS;
      field.particleSpeedKnots = windNinjaSample.speedKt;
      field.particleSpeed = windNinjaSample.speedKt / KNOTS_PER_MPS;
      field.windNinjaRatio = windNinjaSample.ratio;
      field.windNinjaCoverage = windNinjaSample.coverage;
      field.sourceType = "windninja-particle";
    }
    return field;
  }

  windNinjaDataSampleAt(latlng, tileState = this.windNinjaCorsica50mTiles, layerKey = "windninja50") {
    if (!this.visibleLayers[layerKey] || !tileState || tileState.encoding !== "data") return null;
    const step = windNinjaCorsicaStepKey(this, tileState);
    if (!step || !tileState.zooms?.length) return null;
    if (tileState.bounds && !tileState.bounds.contains(latlng)) return null;
    const minZoom = Math.min(...tileState.zooms);
    const maxZoom = Math.max(...tileState.zooms);
    const z = Math.max(minZoom, Math.min(maxZoom, Math.floor(this.map.getZoom())));
    const coords = lonLatToTilePixel(latlng.lng, latlng.lat, z, tileState.tileSize || 256);
    if (!coords) return null;
    const cacheKey = `${step}:${z}:${coords.x}:${coords.y}`;
    const cached = this.windNinjaDataTileCache.get(cacheKey);
    if (!cached) {
      this.loadWindNinjaDataTile(tileState, step, { z, x: coords.x, y: coords.y }, cacheKey);
      return null;
    }
    if (cached.status !== "ready") return null;
    return decodeWindNinjaDataPixel(cached.ctx, coords.px, coords.py);
  }

  loadWindNinjaDataTile(tileState, step, coords, cacheKey) {
    const pending = { status: "loading" };
    this.windNinjaDataTileCache.set(cacheKey, pending);
    const image = new Image();
    image.onload = () => {
      const size = tileState.tileSize || 256;
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(image, 0, 0, size, size);
      this.windNinjaDataTileCache.set(cacheKey, { status: "ready", ctx });
    };
    image.onerror = () => {
      this.windNinjaDataTileCache.set(cacheKey, { status: "error" });
    };
    image.src = windNinjaDataTileUrl(tileState, step, coords);
  }

  targetParticleCount(size) {
    const area = size.x * size.y;
    const zoomFactor = this.map.getZoom() >= 11 ? 1.08 : 0.82;
    return Math.max(18, Math.min(760, Math.round((area / 8200) * zoomFactor * this.particleDensity)));
  }

  seedParticle(size) {
    for (let attempt = 0; attempt < 10; attempt += 1) {
      const x = Math.random() * size.x;
      const y = Math.random() * size.y;
      const field = this.particleFieldAt(this.map.containerPointToLatLng([x, y]));
      if (!field) continue;
      const maxAge = (65 + Math.random() * 95) * this.particleLifeScale;
      return {
        x,
        y,
        age: Math.random() * maxAge,
        maxAge,
        jitter: 0.82 + Math.random() * 0.46,
        liftPhase: Math.random() * Math.PI * 2,
        liftMemory: 0,
        visualAltitude: 0,
        previousVisualAltitude: 0,
      };
    }
    return null;
  }

  animateParticles(timestamp) {
    if (!this.particleCanvas || !this.particleCtx || !this.map || this.particleCanvas.hidden || document.hidden) return;
    if (this.zoomAnimating || this.motionActive || timestamp < this.inputPauseUntil) return;
    if (this.lastParticleDrawTime && timestamp - this.lastParticleDrawTime < 28) return;
    this.lastParticleDrawTime = timestamp;
    const size = this.map.getSize();
    const ctx = this.particleCtx;
    const previous = this.lastParticleTime ?? timestamp;
    const dt = Math.max(0.35, Math.min(2.4, (timestamp - previous) / 16.7));
    this.lastParticleTime = timestamp;
    const targetCount = this.targetParticleCount(size);
    const seedBudget = this.particles.length ? 18 : 42;
    let seeded = 0;
    while (this.particles.length < targetCount && seeded < seedBudget) {
      const particle = this.seedParticle(size);
      if (!particle) break;
      this.particles.push(particle);
      seeded += 1;
    }
    if (this.particles.length > targetCount) this.particles.length = targetCount;

    const metrics = this.renderMetrics || this.overlayRenderMetrics(size);
    ctx.clearRect(-metrics.pad.x, -metrics.pad.y, metrics.cssSize.x, metrics.cssSize.y);
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let index = 0; index < this.particles.length; index += 1) {
      const particle = this.particles[index];
      const latlng = this.map.containerPointToLatLng([particle.x, particle.y]);
      const field = this.particleFieldAt(latlng);
      particle.age += dt;
      if (!field || particle.age > particle.maxAge || particle.x < -24 || particle.x > size.x + 24 || particle.y < -24 || particle.y > size.y + 24) {
        this.particles[index] = this.seedParticle(size) || particle;
        continue;
      }
      const speedKnots = field.particleSpeedKnots ?? field.speedKnots ?? field.speed * KNOTS_PER_MPS;
      const intensity = Math.max(0, Math.min(1, speedKnots / this.scaleMaxKnots));
      const ratioBoost = Math.max(0.45, Math.min(1.9, field.windNinjaRatio || 1));
      const coverage = Math.max(0.18, Math.min(1, field.windNinjaCoverage || 0.62));
      const liftScore = field.windNinjaRatio
        ? smoothstep(1.05, 1.32, ratioBoost) * coverage * (0.35 + intensity * 0.65)
        : 0;
      particle.liftMemory = Math.max(liftScore, (particle.liftMemory || 0) * 0.92);
      particle.previousVisualAltitude = particle.visualAltitude || 0;
      const altitudeTarget = Math.max(0, Math.min(1, particle.liftMemory));
      const altitudeRate = altitudeTarget > particle.visualAltitude ? 0.105 : 0.042;
      particle.visualAltitude += (altitudeTarget - particle.visualAltitude) * altitudeRate * dt;
      particle.visualAltitude = Math.max(0, Math.min(1, particle.visualAltitude));
      const climbRate = particle.visualAltitude - particle.previousVisualAltitude;
      const angle = degToRad(field.flowToDeg);
      const pxSpeed = (0.28 + intensity * 2.35) * ratioBoost * coverage * particle.jitter * dt;
      const previousX = particle.x;
      const previousY = particle.y;
      const flowX = Math.sin(angle);
      const flowY = -Math.cos(angle);
      particle.x += flowX * pxSpeed;
      particle.y += flowY * pxSpeed;
      const life = Math.sin(Math.PI * Math.min(1, particle.age / particle.maxAge));
      const altitude = particle.visualAltitude;
      const descending = climbRate < -0.0015;
      const tailFade = 1 - altitude * 0.62;
      const alpha = (0.1 + intensity * 0.24 + Math.max(0, ratioBoost - 1) * 0.1) * coverage * life * tailFade * this.particleOpacity;
      const baseLength = 2.4 + intensity * 7.2 + Math.max(0, ratioBoost - 1) * 4.8;
      const particleLength = Math.max(0.42, baseLength * (1 - altitude * 0.92) * this.particleSizeScale);
      const lineWidth = Math.max(0.45, (0.62 + intensity * 0.78 - altitude * 0.2 + (descending ? 0.12 : 0)) * Math.sqrt(this.particleSizeScale));
      if (alpha > 0.025) {
        ctx.beginPath();
        ctx.moveTo(particle.x - flowX * particleLength, particle.y - flowY * particleLength);
        ctx.lineTo(particle.x, particle.y);
        ctx.strokeStyle = field.windNinjaRatio
          ? `rgba(226, 246, 255, ${alpha.toFixed(3)})`
          : `rgba(220, 252, 255, ${alpha.toFixed(3)})`;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
      }
    }
    ctx.restore();
  }
}

function smoothstep(edge0, edge1, value) {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function effectiveFieldAlpha(field) {
  return Math.max(0, Math.min(1, field?.renderAlpha ?? 1)) * Math.max(0, Math.min(1, field?.domainFeather ?? 1));
}

function degToRad(value) {
  return (value * Math.PI) / 180;
}

function windDirectionToDeg(u, v) {
  return (Math.atan2(u, v) * 180 / Math.PI + 360) % 360;
}

function haversineMeters(lat1, lon1, lat2, lon2) {
  const radius = 6_371_000;
  const dLat = degToRad(lat2 - lat1);
  const dLon = degToRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(degToRad(lat1)) * Math.cos(degToRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * radius * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function gridValue(grid, row, col, rows, cols) {
  const r = Math.max(0, Math.min(rows - 1, row));
  const c = Math.max(0, Math.min(cols - 1, col));
  return grid[r][c];
}

function nearestGridValue(grid, row, col, rows, cols) {
  if (!Array.isArray(grid)) return null;
  return gridValue(grid, Math.round(row), Math.round(col), rows, cols);
}

function bilinearGrid(grid, row, col, rows, cols) {
  if (!Array.isArray(grid)) return null;
  const r0 = Math.floor(row);
  const c0 = Math.floor(col);
  const r1 = r0 + 1;
  const c1 = c0 + 1;
  const tr = row - r0;
  const tc = col - c0;
  const v00 = gridValue(grid, r0, c0, rows, cols);
  const v10 = gridValue(grid, r1, c0, rows, cols);
  const v01 = gridValue(grid, r0, c1, rows, cols);
  const v11 = gridValue(grid, r1, c1, rows, cols);
  if ([v00, v10, v01, v11].some((value) => value === null || Number.isNaN(value))) return null;
  return (
    v00 * (1 - tr) * (1 - tc) +
    v10 * tr * (1 - tc) +
    v01 * (1 - tr) * tc +
    v11 * tr * tc
  );
}

function buildCfdCorrection(payload) {
  if (!payload || !Array.isArray(payload.lines) || payload.lines.length === 0) return null;
  const samples = payload.lines
    .map((line) => {
      const path = Array.isArray(line.path) ? line.path : [];
      if (!path.length || !Number.isFinite(line.speed_ratio_vs_arome)) return null;
      const lat = path.reduce((sum, point) => sum + Number(point.lat || 0), 0) / path.length;
      const lng = path.reduce((sum, point) => sum + Number(point.lng || 0), 0) / path.length;
      return {
        lat,
        lng,
        ratio: Math.max(0.55, Math.min(1.65, Number(line.speed_ratio_vs_arome))),
        effect: line.effect || "neutral",
      };
    })
    .filter(Boolean);
  if (!samples.length) return null;
  const bounds = samples.reduce(
    (acc, sample) => ({
      minLat: Math.min(acc.minLat, sample.lat),
      maxLat: Math.max(acc.maxLat, sample.lat),
      minLon: Math.min(acc.minLon, sample.lng),
      maxLon: Math.max(acc.maxLon, sample.lng),
    }),
    { minLat: Infinity, maxLat: -Infinity, minLon: Infinity, maxLon: -Infinity }
  );
  return {
    payload,
    samples,
    bounds,
    label: payload.label || "OpenFOAM local CFD",
    targetCellM: payload.target_cell_m || null,
    referenceSpeedMs: payload.reference_arome_speed_ms || null,
  };
}

function buildCoastalTileLayer(payload, sceneId = "ajaccio") {
  const scene = payload?.scenes?.find((candidate) => candidate.sceneId === sceneId);
  if (!scene?.tiles?.length) return null;
  const tiles = scene.tiles.map((tile) => ({
    id: tile.id,
    center: tile.center,
    polygon: tile.strip.polygon,
    priority: Number(tile.priority || 0),
    mesh: tile.mesh || [],
    forcing: tile.forcing || {},
  }));
  const bounds = tiles.reduce(
    (acc, tile) => {
      for (const point of tile.polygon) {
        acc.minLat = Math.min(acc.minLat, point.lat);
        acc.maxLat = Math.max(acc.maxLat, point.lat);
        acc.minLon = Math.min(acc.minLon, point.lng);
        acc.maxLon = Math.max(acc.maxLon, point.lng);
      }
      return acc;
    },
    { minLat: Infinity, maxLat: -Infinity, minLon: Infinity, maxLon: -Infinity }
  );
  return {
    sceneId,
    tileCount: scene.tileCount || tiles.length,
    definition: scene.definition || {},
    tiles,
    bounds,
  };
}

function buildBayModelLayer(payload) {
  if (!payload?.tiles?.length || !Array.isArray(payload.corridor)) return null;
  const tiles = payload.tiles.map((tile) => ({
    id: tile.id,
    rank: Number(tile.rank || 999),
    center: tile.center,
    polygon: tile.strip?.polygon || [],
    score: Number(tile.score || 0),
    fidelityId: tile.fidelity?.id || "surface_downscale_10m",
    fidelityLabel: tile.fidelity?.label || "Downscaling surface",
    matchedHotspots: tile.matched_hotspots || [],
  }));
  const bounds = tiles.reduce(
    (acc, tile) => {
      for (const point of tile.polygon) {
        acc.minLat = Math.min(acc.minLat, point.lat);
        acc.maxLat = Math.max(acc.maxLat, point.lat);
        acc.minLon = Math.min(acc.minLon, point.lng);
        acc.maxLon = Math.max(acc.maxLon, point.lng);
      }
      return acc;
    },
    { minLat: Infinity, maxLat: -Infinity, minLon: Infinity, maxLon: -Infinity }
  );
  return {
    objective: payload.objective || {},
    summary: payload.summary || {},
    corridor: payload.corridor,
    hotspots: payload.hotspots || [],
    tiles,
    bounds,
  };
}

function boundsArrayToObject(bounds) {
  if (!Array.isArray(bounds) || bounds.length !== 4) return null;
  const [minLon, minLat, maxLon, maxLat] = bounds.map(Number);
  if (![minLon, minLat, maxLon, maxLat].every(Number.isFinite)) return null;
  return { minLon, minLat, maxLon, maxLat };
}

function buildMultiscalePlanLayer(payload) {
  const domains = payload?.domains;
  if (!domains?.target_ajaccio_5km_planning_bounds_wgs84) return null;
  return {
    ...payload,
    targetBounds: boundsArrayToObject(domains.target_ajaccio_5km_planning_bounds_wgs84),
    currentBounds: boundsArrayToObject(domains.current_ajaccio_local_bounds_wgs84),
    corsicaBounds: boundsArrayToObject(domains.corsica_arome_bounds_wgs84),
  };
}

function buildLocalWindLayer(payload) {
  if (!payload?.forecast_steps?.length || !payload.domain?.bounds_wgs84) return null;
  const [minLon, minLat, maxLon, maxLat] = payload.domain.bounds_wgs84;
  return {
    ...payload,
    bounds: { minLon, minLat, maxLon, maxLat },
  };
}

function buildRawWindLayer(payload) {
  if (!payload?.forecast_steps?.length || !payload.bbox_wgs84) return null;
  return {
    ...payload,
    forecast_steps: payload.forecast_steps.map((step) => ({
      ...step,
      lead_hour: Number(step.lead_hour),
    })),
  };
}

const RAW_LAYER_KEYS = ["arome", "aromepi", "moloch", "icon2i"];
const OPTIONAL_RAW_LAYER_PRELOAD_ORDER = ["arome", "icon2i", "moloch"];
const RAW_LAYER_LABELS = {
  arome: "AROME",
  aromepi: "AROME-PI",
  moloch: "MOLOCH",
  icon2i: "ICON-2I",
};

function isRawLayerKey(layer) {
  return RAW_LAYER_KEYS.includes(layer);
}

function rawLayerLabel(layer) {
  return RAW_LAYER_LABELS[layer] || "Modèle";
}

function modelPayloadSignature(model) {
  const steps = model?.forecast_steps || [];
  const first = steps[0]?.valid_time_utc || "";
  const last = steps[steps.length - 1]?.valid_time_utc || "";
  return [model?.run_time_utc || "", model?.generated_at_utc || "", steps.length, first, last].join("|");
}

function rawLayerKeyForPayload(payload) {
  const product = String(payload?.product || "").toLowerCase();
  if (product.includes("aromepi") || product.includes("arome-pi")) return "aromepi";
  if (product.includes("moloch")) return "moloch";
  if (product.includes("icon")) return "icon2i";
  if (product.includes("arome")) return "arome";
  return null;
}

function anyRawLayerVisible(overlay) {
  return RAW_LAYER_KEYS.some((layer) => Boolean(overlay.visibleLayers?.[layer]));
}

function firstAvailableRawLayer(overlay) {
  return RAW_LAYER_KEYS.find((layer) => Boolean(rawModelForKey(overlay, layer))) || null;
}

function setFirstAvailableRawLayerVisible(overlay) {
  const fallbackLayer = firstAvailableRawLayer(overlay);
  if (!fallbackLayer) return false;
  for (const rawLayer of RAW_LAYER_KEYS) overlay.visibleLayers[rawLayer] = rawLayer === fallbackLayer;
  return true;
}

function tilePixelToLatLng(globalX, globalY, zoom, tileSize = 256) {
  const worldSize = tileSize * 2 ** zoom;
  const wrappedX = ((globalX % worldSize) + worldSize) % worldSize;
  const clampedY = Math.max(0, Math.min(worldSize, globalY));
  const lng = (wrappedX / worldSize) * 360 - 180;
  const n = Math.PI - (2 * Math.PI * clampedY) / worldSize;
  const lat = (180 / Math.PI) * Math.atan(Math.sinh(n));
  return L.latLng(lat, lng);
}

function prefetchTileRange(map, center, zoom, urlTemplate, cache, maxTiles = 18) {
  if (!map || !center || !Number.isFinite(zoom)) return;
  const tileSize = 256;
  const size = map.getSize();
  const centerPoint = map.project(center, zoom);
  const topLeft = centerPoint.subtract(size.divideBy(2));
  const bottomRight = centerPoint.add(size.divideBy(2));
  const worldTiles = 2 ** zoom;
  const minX = Math.floor(topLeft.x / tileSize) - 1;
  const maxX = Math.floor(bottomRight.x / tileSize) + 1;
  const minY = Math.max(0, Math.floor(topLeft.y / tileSize) - 1);
  const maxY = Math.min(worldTiles - 1, Math.floor(bottomRight.y / tileSize) + 1);
  const centerTileX = Math.floor(centerPoint.x / tileSize);
  const centerTileY = Math.floor(centerPoint.y / tileSize);
  const coords = [];
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const wrappedX = ((x % worldTiles) + worldTiles) % worldTiles;
      coords.push({
        x: wrappedX,
        y,
        distance: Math.abs(x - centerTileX) + Math.abs(y - centerTileY),
      });
    }
  }
  coords.sort((a, b) => a.distance - b.distance);
  for (const { x, y } of coords.slice(0, maxTiles)) {
    const key = `${zoom}:${x}:${y}`;
    if (cache.has(key)) continue;
    const image = new Image();
    image.decoding = "async";
    cache.set(key, image);
    if (cache.size > 420) cache.delete(cache.keys().next().value);
    image.src = urlTemplate.replace("{z}", zoom).replace("{x}", x).replace("{y}", y);
  }
}

function viewportTileCoords(map, center, zoom, tileSize = 256, maxTiles = 18) {
  if (!map || !center || !Number.isFinite(zoom)) return [];
  const size = map.getSize();
  const centerPoint = map.project(center, zoom);
  const topLeft = centerPoint.subtract(size.divideBy(2));
  const bottomRight = centerPoint.add(size.divideBy(2));
  const worldTiles = 2 ** zoom;
  const minX = Math.floor(topLeft.x / tileSize) - 1;
  const maxX = Math.floor(bottomRight.x / tileSize) + 1;
  const minY = Math.max(0, Math.floor(topLeft.y / tileSize) - 1);
  const maxY = Math.min(worldTiles - 1, Math.floor(bottomRight.y / tileSize) + 1);
  const centerTileX = Math.floor(centerPoint.x / tileSize);
  const centerTileY = Math.floor(centerPoint.y / tileSize);
  const coords = [];
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      coords.push({
        z: zoom,
        x: ((x % worldTiles) + worldTiles) % worldTiles,
        y,
        distance: Math.abs(x - centerTileX) + Math.abs(y - centerTileY),
      });
    }
  }
  return coords.sort((a, b) => a.distance - b.distance).slice(0, maxTiles);
}

function heatTileCacheKey(overlay, coords, tileSize) {
  const layer = rawModelKey(overlay);
  const signature = overlay.rawLayerPayloadSignatures?.[layer] || modelPayloadSignature(rawModelForKey(overlay, layer));
  return [
    layer,
    signature,
    overlay.activeLeadHour,
    overlay.displayMode,
    Math.round(overlay.scaleMaxKnots),
    tileSize,
    coords.z,
    coords.x,
    coords.y,
  ].join(":");
}

function cloneCanvas(source) {
  const target = document.createElement("canvas");
  target.width = source.width;
  target.height = source.height;
  target.getContext("2d", { alpha: true }).drawImage(source, 0, 0);
  return target;
}

function createWindHeatTileCanvas(overlay, tileSizePoint) {
  const tileSize = tileSizePoint?.x || 256;
  const ratio = overlay.canvasPixelRatio?.() || Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const tile = document.createElement("canvas");
  tile.className = "wind-heat-tile";
  tile.width = Math.ceil(tileSize * ratio);
  tile.height = Math.ceil(tileSize * ratio);
  tile.style.width = `${tileSize}px`;
  tile.style.height = `${tileSize}px`;
  const ctx = tile.getContext("2d", { alpha: true });
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { tile, ctx, ratio, tileSize };
}

function drawWindHeatTile(tileState, overlay, coords) {
  const { tile, ctx, ratio, tileSize, cacheKey } = tileState;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, tileSize, tileSize);
  if (!overlay.heatLayerVisible?.()) return tile;

  const pad = 44;
  const sampleScale = tileSize < 300 ? 0.34 : 0.26;
  const sourceCssSize = tileSize + pad * 2;
  const renderSize = Math.max(96, Math.ceil(sourceCssSize * sampleScale));
  const source = document.createElement("canvas");
  source.width = renderSize;
  source.height = renderSize;
  const sourceCtx = source.getContext("2d", { alpha: true });
  const image = sourceCtx.createImageData(renderSize, renderSize);
  const worldTileX = coords.x * tileSize;
  const worldTileY = coords.y * tileSize;

  for (let y = 0; y < renderSize; y += 1) {
    for (let x = 0; x < renderSize; x += 1) {
      const tileX = (x / renderSize) * sourceCssSize - pad;
      const tileY = (y / renderSize) * sourceCssSize - pad;
      const latlng = tilePixelToLatLng(worldTileX + tileX, worldTileY + tileY, coords.z, tileSize);
      const field = overlay.fieldAt(latlng);
      const offset = (y * renderSize + x) * 4;
      if (!field) {
        image.data[offset + 3] = 0;
        continue;
      }
      const render = renderFieldColor(field, overlay.displayMode, overlay.scaleMaxKnots);
      const renderAlpha = Math.max(0, Math.min(1, field.renderAlpha ?? 1));
      if (renderAlpha < 0.08) {
        image.data[offset + 3] = 0;
        continue;
      }
      image.data[offset] = render.rgb[0];
      image.data[offset + 1] = render.rgb[1];
      image.data[offset + 2] = render.rgb[2];
      image.data[offset + 3] = Math.round(Math.min(220, render.alpha) * (field.domainFeather ?? 1) * renderAlpha);
    }
  }

  sourceCtx.putImageData(image, 0, 0);
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, -pad, -pad, sourceCssSize, sourceCssSize);
  ctx.restore();
  if (cacheKey) {
    overlay.heatTileRenderCache.set(cacheKey, cloneCanvas(tile));
    if (overlay.heatTileRenderCache.size > 96) {
      overlay.heatTileRenderCache.delete(overlay.heatTileRenderCache.keys().next().value);
    }
  }
  return tile;
}

function drawCachedWindHeatTile(tileState, cached) {
  tileState.ctx.setTransform(1, 0, 0, 1, 0, 0);
  tileState.ctx.clearRect(0, 0, tileState.tile.width, tileState.tile.height);
  tileState.ctx.drawImage(cached, 0, 0);
}

function prewarmWindHeatTiles(overlay, center, zoom, maxTiles = 6) {
  if (!overlay.map || !overlay.heatLayerVisible?.()) return;
  const coordsList = viewportTileCoords(overlay.map, center, zoom, 256, maxTiles);
  const startedAt = performance.now();
  const renderBudget = overlay.zoomAnimating ? 5 : 9;
  const tileBudget = overlay.zoomAnimating ? 2 : 4;
  let rendered = 0;
  for (const coords of coordsList) {
    const tileState = createWindHeatTileCanvas(overlay, L.point(256, 256));
    tileState.cacheKey = heatTileCacheKey(overlay, coords, tileState.tileSize);
    if (overlay.heatTileRenderCache.has(tileState.cacheKey) || overlay.heatTilePrewarmKeys.has(tileState.cacheKey)) continue;
    overlay.heatTilePrewarmKeys.add(tileState.cacheKey);
    try {
      drawWindHeatTile(tileState, overlay, coords);
    } catch (error) {
      console.debug("Wind heat tile prewarm failed", error);
    } finally {
      overlay.heatTilePrewarmKeys.delete(tileState.cacheKey);
    }
    rendered += 1;
    if (rendered >= tileBudget || performance.now() - startedAt > renderBudget) break;
  }
}

function scheduleWindHeatTileRender(tileState, overlay, coords, done) {
  const render = () => {
    if (overlay.zoomAnimating || performance.now() < (overlay.inputPauseUntil || 0)) {
      requestAnimationFrame(render);
      return;
    }
    try {
      const cached = tileState.cacheKey ? overlay.heatTileRenderCache.get(tileState.cacheKey) : null;
      if (cached) {
        drawCachedWindHeatTile(tileState, cached);
        if (done) done(null, tileState.tile);
        return;
      }
      drawWindHeatTile(tileState, overlay, coords);
      if (done) done(null, tileState.tile);
    } catch (error) {
      if (done) done(error, tileState.tile);
    }
  };
  if (window.requestIdleCallback) {
    window.requestIdleCallback(render, { timeout: 220 });
  } else {
    window.setTimeout(render, 0);
  }
}

function createWindHeatTileLayer(overlay) {
  const WindHeatLayer = L.GridLayer.extend({
    createTile(coords, done) {
      const tileState = createWindHeatTileCanvas(overlay, this.getTileSize());
      tileState.cacheKey = heatTileCacheKey(overlay, coords, tileState.tileSize);
      const cached = overlay.heatTileRenderCache.get(tileState.cacheKey);
      if (cached) {
        drawCachedWindHeatTile(tileState, cached);
        if (done) queueMicrotask(() => done(null, tileState.tile));
        return tileState.tile;
      }
      scheduleWindHeatTileRender(tileState, overlay, coords, done);
      return tileState.tile;
    },
    _updateOpacity() {
      if (!this._map) return;
      if (this._fadeFrame) {
        L.Util.cancelAnimFrame(this._fadeFrame);
        this._fadeFrame = null;
      }
      const loading = !this._noTilesToLoad();
      for (const key in this._tiles) {
        const tile = this._tiles[key];
        if (!tile?.el) continue;
        const visible = Boolean(tile.current);
        tile.el.style.opacity = visible ? "1" : "0";
        tile.el.style.visibility = visible ? "visible" : "hidden";
      }
      if (!loading) {
        this._pruneTiles();
      }
    },
  });
  return new WindHeatLayer({
    pane: "windHeatPane",
    tileSize: 256,
    opacity: 1,
    updateWhenZooming: false,
    updateWhenIdle: false,
    keepBuffer: 2,
    className: "wind-heat-grid",
  });
}

function refreshPinnedPointInspector(overlay) {
  overlay?.refreshPinnedPointInspector?.();
}

function forecastStepByLead(model, leadHour) {
  if (!model?.forecast_steps?.length || leadHour === null || leadHour === undefined) return null;
  return model.forecast_steps.find((step) => Number(step.lead_hour) === Number(leadHour)) || null;
}

function forecastIndexByLead(model, leadHour) {
  if (!model?.forecast_steps?.length || leadHour === null || leadHour === undefined) return -1;
  return model.forecast_steps.findIndex((step) => Number(step.lead_hour) === Number(leadHour));
}

function forecastStepByValidTime(model, validTimeUtc) {
  if (!model?.forecast_steps?.length || !validTimeUtc) return null;
  const targetMs = new Date(validTimeUtc).getTime();
  if (!Number.isFinite(targetMs)) return null;
  return (
    model.forecast_steps.find((step) => step.valid_time_utc === validTimeUtc) ||
    model.forecast_steps.find((step) => new Date(step.valid_time_utc).getTime() === targetMs) ||
    null
  );
}

function closestForecastStepByValidTime(model, validTimeUtc) {
  const exact = forecastStepByValidTime(model, validTimeUtc);
  if (exact || !model?.forecast_steps?.length || !validTimeUtc) return exact;
  const targetMs = new Date(validTimeUtc).getTime();
  if (!Number.isFinite(targetMs)) return null;
  return model.forecast_steps.reduce((best, step) => {
    const bestGap = Math.abs(new Date(best.valid_time_utc).getTime() - targetMs);
    const stepGap = Math.abs(new Date(step.valid_time_utc).getTime() - targetMs);
    return stepGap < bestGap ? step : best;
  }, model.forecast_steps[0]);
}

function setActiveLeadHour(overlay, leadHour) {
  overlay.activeLeadHour = Number(leadHour);
  const aromeIndex = forecastIndexByLead(overlay.payload, overlay.activeLeadHour);
  if (aromeIndex >= 0) overlay.stepIndex = aromeIndex;
}

function rawModelKey(overlay) {
  if (overlay.visibleLayers.aromepi && overlay.aromepi) return "aromepi";
  if (overlay.visibleLayers.icon2i && overlay.icon2i) return "icon2i";
  return overlay.visibleLayers.moloch && overlay.moloch ? "moloch" : "arome";
}

function rawModelForKey(overlay, key) {
  if (key === "arome") return overlay.arome;
  if (key === "aromepi") return overlay.aromepi;
  if (key === "icon2i") return overlay.icon2i;
  if (key === "moloch") return overlay.moloch;
  return null;
}

function rawModelUrl(layer) {
  if (layer === "arome") return AROME_DATA_URL;
  if (layer === "aromepi") return AROMEPI_DATA_URL;
  if (layer === "moloch") return MOLOCH_DATA_URL;
  if (layer === "icon2i") return ICON2I_DATA_URL;
  return null;
}

function setRawModelPayload(overlay, layer, payload) {
  const model = buildRawWindLayer(payload);
  if (!model) return false;
  if (layer === "arome") overlay.arome = model;
  if (layer === "aromepi") overlay.aromepi = model;
  if (layer === "moloch") overlay.moloch = model;
  if (layer === "icon2i") overlay.icon2i = model;
  if (overlay.primaryRawLayer === layer) overlay.payload = model;
  overlay.rawLayerPayloadSignatures[layer] = modelPayloadSignature(model);
  return Boolean(rawModelForKey(overlay, layer));
}

async function ensureRawModelLoaded(overlay, layer) {
  if (rawModelForKey(overlay, layer)) return true;
  const url = rawModelUrl(layer);
  if (!url) return false;
  if (overlay.rawLayerLoadPromises?.[layer]) return overlay.rawLayerLoadPromises[layer];
  overlay.rawLayerLoading[layer] = true;
  overlay.rawLayerLoadError[layer] = null;
  syncLayerControls(overlay);
  overlay.rawLayerLoadPromises[layer] = (async () => {
    try {
      const payload = await fetchOptionalJson(url, false, true);
      if (!payload) throw new Error(`Unable to load ${url}`);
      return setRawModelPayload(overlay, layer, payload);
    } catch (error) {
      overlay.rawLayerLoadError[layer] = error;
      console.warn(`Failed to load ${rawLayerLabel(layer)}`, error);
      return false;
    } finally {
      overlay.rawLayerLoading[layer] = false;
      delete overlay.rawLayerLoadPromises[layer];
      syncLayerControls(overlay);
      refreshPinnedPointInspector(overlay);
    }
  })();
  return overlay.rawLayerLoadPromises[layer];
}

async function preloadOptionalRawModels(overlay) {
  if (document.hidden) return;
  const layers = OPTIONAL_RAW_LAYER_PRELOAD_ORDER.filter(
    (layer) => !rawModelForKey(overlay, layer) && !overlay.rawLayerLoadError?.[layer]
  );
  await Promise.allSettled(layers.map((layer) => ensureRawModelLoaded(overlay, layer)));
}

function scheduleOptionalRawModelPreload(overlay) {
  const start = () => {
    preloadOptionalRawModels(overlay).catch((error) => {
      console.debug("Optional raw model preload failed", error);
    });
  };
  const afterFirstPaint = () => {
    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(start, { timeout: 2500 });
    } else {
      window.setTimeout(start, 1200);
    }
  };
  window.requestAnimationFrame(() => window.setTimeout(afterFirstPaint, 350));
}

function activeRawModel(overlay) {
  return rawModelForKey(overlay, rawModelKey(overlay));
}

function activeRawStep(overlay) {
  return forecastStepByLead(activeRawModel(overlay), overlay.activeLeadHour);
}

function displayedValidTimeUtc(overlay) {
  return activeRawStep(overlay)?.valid_time_utc || overlay.step?.valid_time_utc || overlay.aromePiStep?.valid_time_utc || overlay.molochStep?.valid_time_utc || overlay.icon2iStep?.valid_time_utc || null;
}

function equivalentStepForLayer(overlay, layer) {
  const validTimeUtc = displayedValidTimeUtc(overlay);
  if (isRawLayerKey(layer)) return closestForecastStepByValidTime(rawModelForKey(overlay, layer), validTimeUtc);
  return closestForecastStepByValidTime(overlay.payload, validTimeUtc);
}

function buildExpandedWindLayer(payload) {
  if (!payload?.forecast_steps?.length || !payload.domain?.bounds_wgs84) return null;
  const [minLon, minLat, maxLon, maxLat] = payload.domain.bounds_wgs84;
  return {
    ...payload,
    bounds: { minLon, minLat, maxLon, maxLat },
  };
}

function buildSpotGridLayer(payload) {
  if (!payload?.spots?.length) return null;
  const spots = payload.spots
    .map((spot) => {
      if (!spot?.forecast_steps?.length || !spot.bounds_wgs84) return null;
      const [minLon, minLat, maxLon, maxLat] = spot.bounds_wgs84;
      return {
        ...spot,
        bounds: { minLon, minLat, maxLon, maxLat },
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.resolution_m - b.resolution_m);
  if (!spots.length) return null;
  const bounds = spots.reduce(
    (acc, spot) => ({
      minLat: Math.min(acc.minLat, spot.bounds.minLat),
      maxLat: Math.max(acc.maxLat, spot.bounds.maxLat),
      minLon: Math.min(acc.minLon, spot.bounds.minLon),
      maxLon: Math.max(acc.maxLon, spot.bounds.maxLon),
    }),
    { minLat: Infinity, maxLat: -Infinity, minLon: Infinity, maxLon: -Infinity }
  );
  return {
    ...payload,
    spots,
    bounds,
  };
}

function buildRasterTileState(payload) {
  if (!RASTER_ENABLED) return null;
  if (!payload?.urlTemplate || !payload.steps?.length || !payload.modes?.length) return null;
  const steps = payload.steps.map((step) => ({ ...step, key: step.key || `h${String(step.lead_hour).padStart(2, "0")}` }));
  const [minLon, minLat, maxLon, maxLat] = payload.bounds_wgs84 || [];
  return {
    ...payload,
    steps,
    modes: new Set(payload.modes),
    bounds: Number.isFinite(minLon)
      ? L.latLngBounds([
          [minLat, minLon],
          [maxLat, maxLon],
        ])
      : null,
    activeLayer: null,
    activeKey: null,
  };
}

function rasterStepKeyForState(state, leadHour) {
  const match = state?.steps?.find((step) => Number(step.lead_hour) === Number(leadHour));
  return match?.key || `h${String(Number(leadHour || 0)).padStart(2, "0")}`;
}

function rasterStepKey(overlay) {
  return rasterStepKeyForState(overlay.rasterTiles, overlay.activeLeadHour);
}

function rasterMode(overlay) {
  if (overlay.displayMode === "quality") return "quality";
  return overlay.displayMode;
}

function rasterDisplayMinZoom(state) {
  return Math.min(...state.zooms);
}

// Availability against a specific model's tile state — independent of which state is currently
// pinned to overlay.rasterTiles, so heatLayerVisible() can probe the active model directly.
function rasterStateAvailable(overlay, state) {
  if (!state || !overlay.map) return false;
  if (overlay.map.getZoom() < rasterDisplayMinZoom(state)) return false;
  const mode = rasterMode(overlay);
  if (!state.modes.has(mode)) return false;
  const key = rasterStepKeyForState(state, overlay.activeLeadHour);
  return state.steps.some((step) => step.key === key);
}

function activeModelRasterAvailable(overlay) {
  return rasterStateAvailable(overlay, overlay.rasterTilesByModel?.[rawModelKey(overlay)]);
}

function isRasterTileAvailable(overlay) {
  return rasterStateAvailable(overlay, overlay.rasterTiles);
}

function isRasterTileActive(overlay) {
  return Boolean(isRasterTileAvailable(overlay) && overlay.rasterTiles?.activeLayer);
}

function updateRasterTileLayer(overlay) {
  if (!overlay.map) return;
  // Pin overlay.rasterTiles to the active model and tear down any other model's live layer.
  const activeModel = rawModelKey(overlay);
  for (const [model, state] of Object.entries(overlay.rasterTilesByModel || {})) {
    if (model !== activeModel && state.activeLayer) {
      overlay.map.removeLayer(state.activeLayer);
      state.activeLayer = null;
      state.activeKey = null;
    }
  }
  overlay.rasterTiles = overlay.rasterTilesByModel?.[activeModel] || null;
  if (!overlay.rasterTiles) return;

  if (!isRasterTileAvailable(overlay)) {
    if (overlay.rasterTiles.activeLayer) {
      overlay.map.removeLayer(overlay.rasterTiles.activeLayer);
      overlay.rasterTiles.activeLayer = null;
      overlay.rasterTiles.activeKey = null;
    }
    return;
  }
  const mode = rasterMode(overlay);
  const step = rasterStepKey(overlay);
  const activeKey = `${step}:${mode}`;
  if (overlay.rasterTiles.activeKey === activeKey && overlay.rasterTiles.activeLayer) return;
  if (overlay.rasterTiles.activeLayer) overlay.map.removeLayer(overlay.rasterTiles.activeLayer);
  // Version tiles by the model run (stable while the run is unchanged) rather than the per-load
  // DATA_VERSION (Date.now()): combined with immutable Cache-Control on the server, the browser
  // then reuses cached tiles across reloads and pan-backs, and only refetches when a new run
  // publishes (runTimeUtc changes → new URL → new layer is rebuilt by the manifest poll).
  const tileTemplate = overlay.rasterTiles.urlTemplate.replace("{step}", step).replace("{mode}", mode);
  const tileVersion = overlay.rasterTiles.runTimeUtc || DATA_VERSION;
  const url = `${tileTemplate}${tileTemplate.includes("?") ? "&" : "?"}v=${encodeURIComponent(tileVersion)}`;
  overlay.rasterTiles.activeLayer = L.tileLayer(url, {
    bounds: overlay.rasterTiles.bounds || undefined,
    minZoom: rasterDisplayMinZoom(overlay.rasterTiles),
    maxNativeZoom: Math.max(...overlay.rasterTiles.zooms),
    maxZoom: 16,
    opacity: overlay.rasterTiles.opacity ?? 0.86,
    keepBuffer: 4,
    updateWhenZooming: false,
    pane: "windHeatPane",
  }).addTo(overlay.map);
  overlay.rasterTiles.activeKey = activeKey;
}

function windNinjaCorsicaMode(overlay) {
  if (["speed", "devente", "acceleration"].includes(overlay.displayMode)) return overlay.displayMode;
  return null;
}

function windNinjaCorsicaStepKey(overlay, tileState) {
  const lead = overlay.activeLeadHour;
  const match = tileState?.steps?.find((step) => Number(step.lead_hour) === Number(lead));
  return match?.key || null;
}

function isWindNinjaCorsicaTileAvailable(overlay, tileState, layerKey) {
  if (!tileState || !overlay.map) return false;
  if (overlay.visibleLayers?.[layerKey] === false) return false;
  const mode = windNinjaCorsicaMode(overlay);
  if (!mode || !tileState.modes.has(mode)) return false;
  return Boolean(windNinjaCorsicaStepKey(overlay, tileState));
}

function removeWindNinjaCorsicaTileLayer(overlay, tileState) {
  if (!tileState?.activeLayer || !overlay.map) return;
  overlay.map.removeLayer(tileState.activeLayer);
  tileState.activeLayer = null;
  tileState.activeKey = null;
}

function windNinjaDataTileUrl(tileState, step, coords) {
  return versionedDataUrl(
    tileState.urlTemplate
      .replace("{step}", step)
      .replace("{mode}", "data")
      .replace("{z}", coords.z)
      .replace("{x}", coords.x)
      .replace("{y}", coords.y)
  );
}

function lonLatToTilePixel(lon, lat, z, tileSize = 256) {
  if (![lon, lat, z].every(Number.isFinite)) return null;
  const clampedLat = Math.max(-85.05112878, Math.min(85.05112878, lat));
  const scale = 2 ** z;
  const xFloat = ((lon + 180) / 360) * scale;
  const latRad = degToRad(clampedLat);
  const yFloat = ((1 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2) * scale;
  const x = Math.floor(xFloat);
  const y = Math.floor(yFloat);
  return {
    z,
    x,
    y,
    px: Math.max(0, Math.min(tileSize - 1, Math.floor((xFloat - x) * tileSize))),
    py: Math.max(0, Math.min(tileSize - 1, Math.floor((yFloat - y) * tileSize))),
  };
}

function decodeWindNinjaDataPixel(ctx, px, py) {
  const data = ctx.getImageData(px, py, 1, 1).data;
  if (data[3] === 0 || data[2] === 0) return null;
  const speedKt = ((data[0] << 8) + data[1]) / 100;
  const ratio = 0.5 + ((data[2] - 1) / 254) * 1.5;
  const coverage = data[3] / 255;
  if (!Number.isFinite(speedKt) || speedKt <= 0 || coverage < 0.025) return null;
  return { speedKt, ratio, coverage };
}

function windNinjaDataColor(overlay, mode, speedKt, ratio, coverage) {
  const feather = smoothstep(0.04, 0.72, coverage);
  if (mode === "devente") {
    const value = Math.max(0, Math.min(1, (1.05 - ratio) / 0.55));
    const signal = smoothstep(0.02, 0.34, 1 - ratio);
    return {
      rgb: interpolateStops(value, [
        [0, [18, 54, 96]],
        [0.35, [41, 121, 151]],
        [0.68, [148, 163, 184]],
        [1, [248, 250, 252]],
      ]),
      alpha: feather * signal * (42 + Math.pow(value, 0.62) * 174),
    };
  }
  if (mode === "acceleration") {
    const value = Math.max(0, Math.min(1, (ratio - 1) / 0.34));
    const signal = smoothstep(0.02, 0.34, ratio - 1);
    return {
      rgb: interpolateStops(value, [
        [0, [19, 78, 74]],
        [0.35, [34, 197, 180]],
        [0.72, [245, 202, 66]],
        [1, [245, 139, 42]],
      ]),
      alpha: feather * signal * (38 + Math.pow(value, 0.58) * 174),
    };
  }
  const intensity = Math.max(0, Math.min(1, speedKt / overlay.scaleMaxKnots));
  return {
    rgb: colorArray(intensity),
    alpha: feather * (54 + Math.pow(intensity, 0.68) * 138),
  };
}

function renderWindNinjaDataImage(overlay, sourceImage, canvas, mode) {
  const width = sourceImage.width || 256;
  const height = sourceImage.height || 256;
  canvas.width = width;
  canvas.height = height;
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const sourceCtx = sourceCanvas.getContext("2d", { willReadFrequently: true });
  sourceCtx.drawImage(sourceImage, 0, 0, width, height);
  const source = sourceCtx.getImageData(0, 0, width, height);
  const target = sourceCtx.createImageData(width, height);
  const data = source.data;
  const output = target.data;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0 || data[i + 2] === 0) {
      output[i + 3] = 0;
      continue;
    }
    const speedKt = ((data[i] << 8) + data[i + 1]) / 100;
    const ratio = 0.5 + ((data[i + 2] - 1) / 254) * 1.5;
    const coverage = data[i + 3] / 255;
    const color = windNinjaDataColor(overlay, mode, speedKt, ratio, coverage);
    output[i] = color.rgb[0];
    output[i + 1] = color.rgb[1];
    output[i + 2] = color.rgb[2];
    output[i + 3] = Math.max(0, Math.min(225, Math.round(color.alpha)));
  }
  canvas.getContext("2d").putImageData(target, 0, 0);
}

function createWindNinjaDataLayer(overlay, tileState, step, mode, zIndex) {
  const layer = L.gridLayer({
    bounds: tileState.bounds || undefined,
    minZoom: Math.min(...tileState.zooms),
    maxNativeZoom: Math.max(...tileState.zooms),
    maxZoom: 16,
    opacity: Number.isFinite(tileState.opacity) ? Math.min(0.86, Math.max(0.5, tileState.opacity)) : 0.76,
    pane: "overlayPane",
    zIndex,
    tileSize: tileState.tileSize || 256,
  });
  layer.createTile = (coords, done) => {
    const canvas = document.createElement("canvas");
    canvas.width = tileState.tileSize || 256;
    canvas.height = tileState.tileSize || 256;
    canvas.className = "windninja-data-tile";
    const image = new Image();
    image.onload = () => {
      try {
        renderWindNinjaDataImage(overlay, image, canvas, mode);
        done(null, canvas);
      } catch (error) {
        done(error, canvas);
      }
    };
    image.onerror = () => done(null, canvas);
    image.src = windNinjaDataTileUrl(tileState, step, coords);
    return canvas;
  };
  return layer;
}

function updateSingleWindNinjaCorsicaTileLayer(overlay, tileState, layerKey, zIndex) {
  if (!overlay.map || !tileState) return;
  if (!isWindNinjaCorsicaTileAvailable(overlay, tileState, layerKey)) {
    removeWindNinjaCorsicaTileLayer(overlay, tileState);
    return;
  }
  const mode = windNinjaCorsicaMode(overlay);
  const step = windNinjaCorsicaStepKey(overlay, tileState);
  const activeKey = `${step}:${mode}`;
  if (tileState.activeKey === activeKey && tileState.activeLayer) return;
  removeWindNinjaCorsicaTileLayer(overlay, tileState);
  if (tileState.encoding === "data") {
    tileState.activeLayer = createWindNinjaDataLayer(overlay, tileState, step, mode, zIndex).addTo(overlay.map);
    tileState.activeKey = activeKey;
    return;
  }
  const url = versionedDataUrl(tileState.urlTemplate.replace("{step}", step).replace("{mode}", mode));
  const opacity = Number.isFinite(tileState.opacity) ? tileState.opacity : 0.66;
  tileState.activeLayer = L.tileLayer(url, {
    bounds: tileState.bounds || undefined,
    minZoom: Math.min(...tileState.zooms),
    maxNativeZoom: Math.max(...tileState.zooms),
    maxZoom: 16,
    opacity: Math.min(0.72, Math.max(0.48, opacity)),
    pane: "overlayPane",
    zIndex,
  }).addTo(overlay.map);
  tileState.activeKey = activeKey;
}

function updateWindNinjaCorsicaTileLayer(overlay) {
  updateSingleWindNinjaCorsicaTileLayer(overlay, overlay.windNinjaCorsica50mTiles, "windninja50", 420);
}

function buildPriorityCorridorLayer(payload) {
  if (!payload?.corridors?.length) return null;
  const corridors = payload.corridors
    .map((corridor) => {
      if (!corridor?.forecast_steps?.length || !corridor.bounds_wgs84) return null;
      const [minLon, minLat, maxLon, maxLat] = corridor.bounds_wgs84;
      return {
        ...corridor,
        bounds: { minLon, minLat, maxLon, maxLat },
      };
    })
    .filter(Boolean);
  if (!corridors.length) return null;
  const bounds = corridors.reduce(
    (acc, corridor) => ({
      minLat: Math.min(acc.minLat, corridor.bounds.minLat),
      maxLat: Math.max(acc.maxLat, corridor.bounds.maxLat),
      minLon: Math.min(acc.minLon, corridor.bounds.minLon),
      maxLon: Math.max(acc.maxLon, corridor.bounds.maxLon),
    }),
    { minLat: Infinity, maxLat: -Infinity, minLon: Infinity, maxLon: -Infinity }
  );
  return {
    ...payload,
    corridors,
    bounds,
  };
}

function buildPriorityCorridorManifest(payload) {
  if (!payload?.corridors?.length) return null;
  const corridors = payload.corridors
    .map((corridor) => {
      if (!corridor?.client_url || !corridor.center || !corridor.bounds_wgs84) return null;
      const [minLon, minLat, maxLon, maxLat] = corridor.bounds_wgs84;
      return {
        ...corridor,
        bounds: { minLon, minLat, maxLon, maxLat },
      };
    })
    .filter(Boolean);
  return corridors.length ? { ...payload, corridors } : null;
}

function nearestPriorityCorridorMeta(manifest, center) {
  if (!manifest?.corridors?.length || !center) return null;
  return manifest.corridors
    .map((corridor) => ({
      corridor,
      distance: haversineMeters(center.lat, center.lng, corridor.center.lat, corridor.center.lon),
    }))
    .filter((item) => item.distance < 18_000)
    .sort((a, b) => a.distance - b.distance)[0]?.corridor || null;
}

function buildValidationState(payload) {
  if (!payload) {
    return {
      status: "missing_casebook",
      matched: 0,
      required: 10,
      factor: 0.35,
      label: "Non chargée",
      badge: "Validation absente",
    };
  }
  const matched = Number(payload.matched_case_count || 0);
  const required = Math.max(1, Number(payload.minimum_product_grade_cases || 10));
  const progress = Math.max(0, Math.min(1, matched / required));
  const factor = payload.status === "product_grade_candidate" ? 1 : 0.35 + progress * 0.45;
  const label = payload.status === "product_grade_candidate" ? `Validé ${matched}/${required}` : `Terrain ${matched}/${required}`;
  return {
    ...payload,
    matched,
    required,
    progress,
    factor,
    label,
    badge: payload.status === "product_grade_candidate" ? "Validation OK" : `Validation ${matched}/${required}`,
  };
}

function buildRegimeQaState(payload) {
  if (!payload?.summary) {
    return {
      status: "missing_regime_qa",
      label: "QA absente",
      badge: "QA priors --",
      title: "Rapport QA des priors spot/régime non chargé.",
    };
  }
  const fixRequired = Number(payload.summary.fix_required_count || 0);
  const review = Number(payload.summary.review_count || 0);
  const ok = Number(payload.summary.priorities?.ok || 0);
  const reference = Number(payload.summary.priorities?.reference || 0);
  const total = ok + reference + fixRequired + review;
  if (fixRequired > 0 || review > 0) {
    const firstIssue = payload.summary.fix_required?.[0] || payload.summary.review?.[0];
    const issueLabel = firstIssue ? `${firstIssue.spot_id} ${firstIssue.regime_id}` : "prior à revoir";
    return {
      ...payload,
      status: fixRequired > 0 ? "fix_required" : "review_required",
      label: `${fixRequired} fix / ${review} revue`,
      badge: `QA ${fixRequired}/${review}`,
      title: `QA priors: ${issueLabel}. Contrat modèle à corriger avant confiance produit.`,
    };
  }
  return {
    ...payload,
    status: "ok",
    label: `Priors OK ${ok}/${total}`,
    badge: "QA priors OK",
    title: "QA priors spot/régime OK. Ce n'est pas une validation terrain.",
  };
}

function buildValidationGapState(payload) {
  if (!payload?.gaps?.length) {
    return {
      status: "missing_validation_gaps",
      missingDecisionCases: 0,
      gaps: [],
      title: "Plan terrain absent",
      detail: "Aucun plan de collecte chargé.",
    };
  }
  const gaps = payload.gaps
    .filter((gap) => Number(gap.needed_cases || 0) > 0)
    .sort((a, b) => Number(a.priority || 999) - Number(b.priority || 999));
  return {
    ...payload,
    missingDecisionCases: Number(payload.missing_decision_cases || 0),
    gaps,
  };
}

function buildFieldTestPacketState(payload) {
  if (!payload?.opportunities?.length) {
    return {
      status: "missing_field_test_packet",
      summary: null,
      opportunities: [],
      title: "Paquet terrain absent",
    };
  }
  const statusRank = {
    available_in_loaded_forecast: 0,
    loaded_forecast_match_past: 1,
    wait_for_regime: 2,
    covered: 3,
  };
  const opportunities = payload.opportunities
    .filter((item) => Number(item.needed_cases || 0) > 0)
    .sort((a, b) => (statusRank[a.status] ?? 99) - (statusRank[b.status] ?? 99) || Number(a.priority || 999) - Number(b.priority || 999));
  const counts = opportunities.reduce(
    (acc, item) => {
      acc[item.status] = (acc[item.status] || 0) + Number(item.needed_cases || 0);
      return acc;
    },
    {}
  );
  return {
    ...payload,
    status: "loaded",
    opportunities,
    counts,
    title: "Paquet terrain chargé. C'est un plan de collecte, pas une validation.",
  };
}

function spotShortLabel(spotId) {
  const labels = {
    ricanto: "Ricanto",
    porticcio: "Porticcio",
    capo_di_feno: "Capo",
    lava: "Lava",
    pointe_castagne: "Castagne",
  };
  return labels[spotId] || spotId || "spot";
}

function regimeShortLabel(regimeId) {
  const labels = {
    sw_wsw: "SW/WSW",
    w_wnw: "W/WNW",
    nw_nnw: "NW/NNW",
    s_sse: "S/SSE",
    e_ne: "E/NE",
  };
  return labels[regimeId] || regimeId || "régime";
}

function pickValidationGap(validationGaps, overlay) {
  if (!validationGaps?.gaps?.length) return null;
  const activeRegimeId = overlay.localStep?.active_regime?.id;
  const activeLeadHour = overlay.localStep?.lead_hour;
  const candidates = validationGaps.gaps;
  return (
    candidates.find(
      (gap) =>
        gap.regime_id === activeRegimeId &&
        gap.current_forecast &&
        Number(gap.current_forecast.lead_hour) === Number(activeLeadHour)
    ) ||
    candidates.find((gap) => gap.regime_id === activeRegimeId && gap.current_forecast_available) ||
    candidates.find((gap) => gap.current_forecast_available) ||
    candidates[0]
  );
}

function pickFieldTestOpportunity(packet, overlay) {
  if (!packet?.opportunities?.length) return null;
  const activeRegimeId = overlay.localStep?.active_regime?.id;
  const activeLeadHour = overlay.localStep?.lead_hour;
  const opportunities = packet.opportunities;
  return (
    opportunities.find(
      (item) =>
        item.regime_id === activeRegimeId &&
        item.forecast &&
        Number(item.forecast.lead_hour) === Number(activeLeadHour)
    ) ||
    opportunities.find((item) => item.regime_id === activeRegimeId) ||
    opportunities[0]
  );
}

function fieldTestStatusLabel(status) {
  if (status === "available_in_loaded_forecast") return "fenêtre fraîche";
  if (status === "loaded_forecast_match_past") return "run chargé périmé";
  if (status === "wait_for_regime") return "attendre régime";
  return "plan terrain";
}

function applyValidationConfidence(value, validation) {
  if (!validation) return value;
  return Math.max(0.08, Math.min(value, value * validation.factor + 0.06));
}

function applyValidationQuality(value, validation) {
  if (!validation) return value;
  if (validation.status === "product_grade_candidate") return value;
  return Math.max(0, Math.min(1, value * validation.factor));
}

function bayTierStyle(fidelityId) {
  if (fidelityId === "openfoam_micro50_candidate") {
    return { rgb: "248, 250, 252", fill: 0.17, stroke: 0.9, width: 2.2 };
  }
  if (fidelityId === "coastal_cfd_2m_candidate") {
    return { rgb: "103, 232, 249", fill: 0.12, stroke: 0.72, width: 1.55 };
  }
  return { rgb: "34, 197, 180", fill: 0.07, stroke: 0.35, width: 0.9 };
}

function colorArray(ratio) {
  const stops = [
    [0, [32, 85, 180]],
    [0.2, [37, 137, 210]],
    [0.38, [34, 197, 180]],
    [0.52, [82, 190, 96]],
    [0.68, [245, 202, 66]],
    [0.82, [245, 139, 42]],
    [0.94, [226, 54, 54]],
    [1, [150, 67, 190]],
  ];
  if (ratio <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i += 1) {
    const [value, rgb] = stops[i];
    const [previousValue, previousRgb] = stops[i - 1];
    if (ratio <= value) {
      const t = (ratio - previousValue) / (value - previousValue);
      return rgb.map((channel, index) => Math.round(previousRgb[index] + (channel - previousRgb[index]) * t));
    }
  }
  return stops.at(-1)[1];
}

function renderFieldColor(field, mode, scaleMaxKnots) {
  if (mode === "surface") {
    const surfaceClass = SURFACE_CLASSES[field.surfaceClassId] || SURFACE_CLASSES[0];
    const confidence = Math.max(0, Math.min(1, field.surfaceConfidence || 0));
    return {
      rgb: surfaceClass.rgb,
      alpha: 82 + confidence * 128,
    };
  }
  if (mode === "devente") {
    const value = Math.max(0, Math.min(1, field.devente || 0));
    return {
      rgb: interpolateStops(value, [
        [0, [18, 54, 96]],
        [0.35, [41, 121, 151]],
        [0.68, [148, 163, 184]],
        [1, [248, 250, 252]],
      ]),
      alpha: 58 + Math.pow(value, 0.62) * 178,
    };
  }
  if (mode === "acceleration") {
    const value = Math.max(0, Math.min(1, field.acceleration || 0));
    return {
      rgb: interpolateStops(value, [
        [0, [19, 78, 74]],
        [0.35, [34, 197, 180]],
        [0.72, [245, 202, 66]],
        [1, [245, 139, 42]],
      ]),
      alpha: 50 + Math.pow(value, 0.58) * 180,
    };
  }
  if (mode === "confidence") {
    const value = Math.max(0, Math.min(1, field.confidence || 0));
    return {
      rgb: interpolateStops(value, [
        [0, [127, 29, 29]],
        [0.36, [245, 139, 42]],
        [0.68, [82, 190, 96]],
        [1, [103, 232, 249]],
      ]),
      alpha: 70 + Math.pow(value, 0.8) * 150,
    };
  }
  if (mode === "quality") {
    const value = Math.max(0, Math.min(1, field.quality || 0));
    const sessionClass = SESSION_CLASSES[field.sessionClassId];
    if (sessionClass) {
      if (field.sessionClassId === 7) {
        return {
          rgb: sessionClass.rgb,
          alpha: 8 + value * 28,
        };
      }
      return {
        rgb: sessionClass.rgb,
        alpha: 74 + Math.pow(Math.max(value, sessionClass.priority), 0.7) * 152,
      };
    }
    return {
      rgb: interpolateStops(value, [
        [0, [15, 23, 42]],
        [0.28, [37, 99, 235]],
        [0.52, [34, 197, 180]],
        [0.76, [245, 202, 66]],
        [1, [248, 250, 252]],
      ]),
      alpha: 46 + Math.pow(value, 0.7) * 188,
    };
  }
  const speedKnots = field.speedKnots ?? field.speed * KNOTS_PER_MPS;
  const intensity = Math.max(0, Math.min(1, speedKnots / scaleMaxKnots));
  return {
    rgb: colorArray(intensity),
    alpha: 84 + Math.pow(intensity, 0.68) * 146,
  };
}

function interpolateStops(value, stops) {
  if (value <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i += 1) {
    const [stopValue, rgb] = stops[i];
    const [previousStopValue, previousRgb] = stops[i - 1];
    if (value <= stopValue) {
      const t = (value - previousStopValue) / (stopValue - previousStopValue);
      return rgb.map((channel, index) => Math.round(previousRgb[index] + (channel - previousRgb[index]) * t));
    }
  }
  return stops.at(-1)[1];
}

function formatHour(iso) {
  const date = new Date(iso);
  return date.toLocaleString("fr-FR", {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: DISPLAY_TIME_ZONE,
  });
}

function formatClock(iso) {
  return new Date(iso).toLocaleString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: DISPLAY_TIME_ZONE,
  });
}

function formatForecastDay(iso) {
  return new Date(iso).toLocaleString("fr-FR", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    timeZone: DISPLAY_TIME_ZONE,
  });
}

function formatRunStamp(iso) {
  return new Date(iso).toLocaleString("fr-FR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: DISPLAY_TIME_ZONE,
  });
}

function formatLeadLabel(step) {
  const minutes = Number.isFinite(Number(step?.lead_minutes))
    ? Number(step.lead_minutes)
    : Math.round(Number(step?.lead_hour || 0) * 60);
  if (minutes % 60 === 0) return `H+${minutes / 60}`;
  if (minutes < 60) return `+${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return `+${hours}h${String(rest).padStart(2, "0")}`;
}

function formatDayNavParts(iso) {
  const parts = new Intl.DateTimeFormat("fr-FR", {
    timeZone: DISPLAY_TIME_ZONE,
    day: "2-digit",
    month: "short",
  }).formatToParts(new Date(iso));
  return {
    day: parts.find((part) => part.type === "day")?.value || "--",
    month: (parts.find((part) => part.type === "month")?.value || "").replace(".", ""),
  };
}

function localDateParts(iso) {
  const parts = new Intl.DateTimeFormat("fr-FR", {
    timeZone: DISPLAY_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(iso));
  const value = (type) => parts.find((part) => part.type === type)?.value || "00";
  return {
    year: value("year"),
    month: value("month"),
    day: value("day"),
    hour: Number(value("hour")),
    minute: Number(value("minute")),
  };
}

function localDayKey(iso) {
  const parts = localDateParts(iso);
  return `${parts.year}-${parts.month}-${parts.day}`;
}

function localTimeMinutes(iso) {
  const parts = localDateParts(iso);
  return parts.hour * 60 + parts.minute;
}

function forecastDayGroups(steps) {
  const groups = new Map();
  for (const step of steps || []) {
    if (!step?.valid_time_utc) continue;
    const key = localDayKey(step.valid_time_utc);
    if (!groups.has(key)) groups.set(key, { key, steps: [], firstMs: Infinity, lastMs: -Infinity });
    const group = groups.get(key);
    const ms = new Date(step.valid_time_utc).getTime();
    group.steps.push(step);
    group.firstMs = Math.min(group.firstMs, ms);
    group.lastMs = Math.max(group.lastMs, ms);
  }
  return [...groups.values()]
    .map((group) => ({
      ...group,
      steps: group.steps.sort((a, b) => new Date(a.valid_time_utc).getTime() - new Date(b.valid_time_utc).getTime()),
    }))
    .sort((a, b) => a.firstMs - b.firstMs);
}

function closestStepInDay(day, referenceStep) {
  const steps = day?.steps || [];
  if (!steps.length) return null;
  if (!referenceStep?.valid_time_utc) return steps[0];
  const targetMinutes = localTimeMinutes(referenceStep.valid_time_utc);
  return steps.reduce((best, step) => {
    const bestGap = Math.abs(localTimeMinutes(best.valid_time_utc) - targetMinutes);
    const stepGap = Math.abs(localTimeMinutes(step.valid_time_utc) - targetMinutes);
    return stepGap < bestGap ? step : best;
  }, steps[0]);
}

function hasWindNinja50Step(overlay, leadHour) {
  return Boolean(
    overlay.windNinjaCorsica50mTiles?.steps?.some((step) => Number(step.lead_hour) === Number(leadHour))
  );
}

function windNinjaModesAvailable(overlay) {
  return Boolean(overlay.visibleLayers.windninja50 && hasWindNinja50Step(overlay, overlay.activeLeadHour));
}

function applyPreferredForecastLayer(overlay) {
  const windNinjaAvailable = hasWindNinja50Step(overlay, overlay.activeLeadHour);
  if (windNinjaAvailable) {
    overlay.visibleLayers.windninja50 = true;
    for (const rawLayer of RAW_LAYER_KEYS) overlay.visibleLayers[rawLayer] = false;
  } else {
    overlay.visibleLayers.windninja50 = false;
    if (!anyRawLayerVisible(overlay)) setFirstAvailableRawLayerVisible(overlay);
  }
  if (!windNinjaAvailable) overlay.displayMode = "speed";
  overlay.heatDirty = true;
  overlay.windNinjaDataTileCache?.clear();
  overlay.syncCanvasVisibility?.();
  overlay.syncParticleVisibility?.();
  overlay.redrawHeatLayer?.();
  syncLayerControls(overlay);
  syncModeControls(overlay);
  refreshActiveLayerLabel(overlay);
  refreshCoverageStatus(overlay);
}

function windNinjaManifestSignature(tileState) {
  if (!tileState?.steps?.length) return "";
  const steps = tileState.steps.map((step) => `${step.key}:${step.lead_hour}`).join("|");
  return `${tileState.tileCount || 0}:${tileState.generatedAt || tileState.generated_at_utc || ""}:${steps}`;
}

async function refreshWindNinja50mManifest(payload, overlay) {
  const manifestPayload =
    (await fetchOptionalJson(WINDNINJA_CORSICA_50M_DATA_MANIFEST_URL, true)) ||
    (await fetchOptionalJson(WINDNINJA_CORSICA_50M_TILES_MANIFEST_URL, true));
  const nextState = buildRasterTileState(manifestPayload);
  if (!nextState) return false;
  if (windNinjaManifestSignature(nextState) === windNinjaManifestSignature(overlay.windNinjaCorsica50mTiles)) return false;

  removeWindNinjaCorsicaTileLayer(overlay, overlay.windNinjaCorsica50mTiles);
  overlay.windNinjaCorsica50mTiles = nextState;
  overlay.windNinjaDataTileCache.clear();
  applyPreferredForecastLayer(overlay);
  overlay.refreshTileLayers();
  buildForecastButtons(payload, overlay);
  refreshActiveLayerLabel(overlay);
  refreshCoverageStatus(overlay);
  return true;
}

function rasterManifestSignature(state) {
  if (!state) return "";
  const steps = (state.steps || [])
    .map((step) => `${step.key || ""}:${step.lead_hour ?? ""}:${step.lead_minutes ?? ""}:${step.valid_time_utc || ""}`)
    .join("|");
  return `${state.model || ""}:${state.runTimeUtc || state.generatedAt || ""}:${state.tileFormat || ""}:${state.tileCount || 0}:${steps}`;
}

// Load the pre-baked colour-tile manifest for each raw model (./tiles/<model>/manifest.json).
// Missing manifests are fine — that model simply falls back to the live JS heat overlay.
async function loadModelRasterManifests(overlay) {
  if (!RASTER_ENABLED) return false;
  let changed = false;
  await Promise.all(
    RAW_LAYER_KEYS.map(async (model) => {
      const payload = await fetchOptionalJson(modelRasterManifestUrl(model), true);
      const state = buildRasterTileState(payload);
      if (!state) return;
      if (rasterManifestSignature(state) === rasterManifestSignature(overlay.rasterTilesByModel[model])) return;
      const previous = overlay.rasterTilesByModel[model];
      if (previous?.activeLayer && overlay.map) overlay.map.removeLayer(previous.activeLayer);
      overlay.rasterTilesByModel[model] = state;
      changed = true;
    })
  );
  if (changed) {
    overlay.refreshTileLayers();
    overlay.redrawHeatLayer();
    refreshCoverageStatus(overlay);
  }
  return changed;
}

function startProgressiveWindNinjaPolling(payload, overlay) {
  const poll = async () => {
    if (document.hidden) return;
    try {
      await loadModelRasterManifests(overlay);
    } catch (error) {
      console.debug("Model raster manifest refresh failed", error);
    }
    try {
      await refreshWindNinja50mManifest(payload, overlay);
    } catch (error) {
      console.debug("WindNinja progressive manifest refresh failed", error);
    }
  };
  window.setInterval(poll, 45000);
}

function showModelUpdateToast(message) {
  const toast = document.querySelector("#model-update-toast");
  if (!toast) return;
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showModelUpdateToast.hideTimer);
  showModelUpdateToast.hideTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 6200);
}

async function fetchModelResourceSignature(url) {
  const candidates = [gzipJsonUrl(url), url].filter(Boolean);
  for (const candidate of candidates) {
    try {
      const response = await fetch(cacheBustedUrl(candidate), { method: "HEAD", cache: "no-store" });
      if (!response.ok) continue;
      return [
        response.headers.get("Last-Modified") || "",
        response.headers.get("Content-Length") || "",
        response.headers.get("ETag") || "",
        response.headers.get("Content-Encoding") || "",
      ].join("|");
    } catch {
      // Try the next candidate, then let the caller keep the current model.
    }
  }
  return null;
}

function redrawAfterModelUpdate(overlay, payload) {
  overlay.heatDirty = true;
  overlay.windNinjaDataTileCache?.clear();
  overlay.resetParticles?.();
  overlay.refreshTileLayers?.();
  overlay.redrawHeatLayer?.();
  syncLayerControls(overlay);
  syncModeControls(overlay);
  buildForecastButtons(overlay.payload || payload, overlay);
  updateReadout(overlay.payload || payload, overlay);
  refreshPinnedPointInspector(overlay);
}

function applyRawModelUpdate(overlay, layer, payload) {
  const activeLayer = rawModelKey(overlay);
  const wasActive = activeLayer === layer;
  const previousValidTime = displayedValidTimeUtc(overlay);
  const previousLeadHour = overlay.activeLeadHour;
  const previousSignature = overlay.rawLayerPayloadSignatures[layer] || modelPayloadSignature(rawModelForKey(overlay, layer));
  if (!setRawModelPayload(overlay, layer, payload)) return false;
  const model = rawModelForKey(overlay, layer);
  const nextSignature = modelPayloadSignature(model);
  if (nextSignature === previousSignature) return false;
  if (wasActive) {
    const nextStep =
      forecastStepByValidTime(model, previousValidTime) ||
      forecastStepByLead(model, previousLeadHour) ||
      closestForecastStepByValidTime(model, previousValidTime) ||
      model.forecast_steps?.[0] ||
      null;
    if (nextStep) setActiveLeadHour(overlay, nextStep.lead_hour);
    redrawAfterModelUpdate(overlay, overlay.payload);
  } else {
    syncLayerControls(overlay);
    refreshCoverageStatus(overlay);
  }
  return true;
}

function startRawModelUpdatePolling(overlay) {
  const poll = async () => {
    if (document.hidden) return;
    for (const layer of RAW_LAYER_KEYS) {
      const url = rawModelUrl(layer);
      if (!url) continue;
      const signature = await fetchModelResourceSignature(url);
      const previousResourceSignature = Object.prototype.hasOwnProperty.call(overlay.rawLayerResourceSignatures, layer)
        ? overlay.rawLayerResourceSignatures[layer]
        : undefined;
      overlay.rawLayerResourceSignatures[layer] = signature;
      if (!signature || previousResourceSignature === undefined || previousResourceSignature === signature) continue;

      const isActive = rawModelKey(overlay) === layer;
      const isLoaded = Boolean(rawModelForKey(overlay, layer));
      if (!isActive && !isLoaded) {
        showModelUpdateToast(`${rawLayerLabel(layer)} disponible`);
        syncLayerControls(overlay);
        continue;
      }

      const payload = await fetchJsonWithGzipFallback(url, true);
      if (!payload?.forecast_steps?.length) continue;
      const changed = applyRawModelUpdate(overlay, layer, payload);
      if (changed && isActive) {
        showModelUpdateToast(`${rawLayerLabel(layer)} mis à jour · run ${formatRunStamp(payload.run_time_utc)}`);
      }
    }
  };
  window.setInterval(() => {
    poll().catch((error) => console.debug("Raw model update polling failed", error));
  }, MODEL_UPDATE_POLL_INTERVAL_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) poll().catch((error) => console.debug("Raw model update polling failed", error));
  });
}

function chooseInitialForecastIndex(payload) {
  const steps = payload.forecast_steps || [];
  const now = Date.now();
  const nextIndex = steps.findIndex((step) => new Date(step.valid_time_utc).getTime() >= now);
  if (nextIndex >= 0) return nextIndex;
  return Math.max(0, steps.length - 1);
}

function updateReadout(payload, overlay) {
  refreshActiveLayerLabel(overlay);
  const model = activeRawModel(overlay) || payload;
  const step = activeRawStep(overlay);
  document.querySelector("#forcing").textContent = model.model_label;
  document.querySelector("#solver").textContent = new Date(model.run_time_utc || payload.run_time_utc).toLocaleString("fr-FR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Europe/Paris",
  });
  document.querySelector("#wind-regime").textContent = `${rawLayerLabel(rawModelKey(overlay))} brut`;
  document.querySelector("#validation-status").textContent = "WN 50 m";
  if (!step) {
    document.querySelector("#spot-speed").textContent = "--";
    document.querySelector("#spot-detail").textContent = `${model.model_label} hors échéance H+${overlay.activeLeadHour}`;
    updateLegendTitle(overlay.displayMode);
    refreshCoverageStatus(overlay);
    return;
  }
  const renderMeanKnots = step.stats_ms.mean * KNOTS_PER_MPS;
  const renderMaxKnots = step.stats_ms.max * KNOTS_PER_MPS;
  const gustMeanKnots = step.gust_stats_ms ? step.gust_stats_ms.mean * KNOTS_PER_MPS : null;
  const gustMaxKnots = step.gust_stats_ms ? step.gust_stats_ms.max * KNOTS_PER_MPS : null;
  const gustLabel = model.gust_window_label || (model.product === "aromepi" ? "rafales 15 min" : "rafales modele");
  document.querySelector("#spot-speed").textContent = `${renderMeanKnots.toFixed(0)} kt`;
  document.querySelector("#spot-detail").textContent = step.gust_stats_ms
    ? `Vent moyen Corse ${renderMeanKnots.toFixed(0)} kt · ${gustLabel} moy. ${gustMeanKnots.toFixed(0)} kt · max ${gustMaxKnots.toFixed(0)} kt · ${formatHour(step.valid_time_utc)}`
    : `Moyenne Corse ${model.model_label} · max ${renderMaxKnots.toFixed(0)} kt · ${formatHour(step.valid_time_utc)}`;
  updateLegendTitle(overlay.displayMode);
  refreshCoverageStatus(overlay);
}

function refreshActiveLayerLabel(overlay) {
  const label = document.querySelector("#layer-height");
  if (!label) return;
  const wn50Resolution = overlay.windNinjaCorsica50mTiles?.source?.resolution_m || 50;
  const wn50Height = overlay.windNinjaCorsica50mTiles?.source?.output_height_m || 10;
  const activeLayers = [
    overlay.visibleLayers.arome ? "AROME" : null,
    overlay.visibleLayers.aromepi ? "AROME-PI" : null,
    overlay.visibleLayers.moloch ? "MOLOCH" : null,
    overlay.visibleLayers.icon2i ? "ICON-2I" : null,
    overlay.visibleLayers.windninja50 ? `WN ${wn50Resolution} m / ${wn50Height} m` : null,
  ].filter(Boolean);
  label.textContent = activeLayers.length ? activeLayers.join(" + ") : "Aucune";
}

function updateValidationUi(validation) {
  const status = document.querySelector("#validation-status");
  const badge = document.querySelector("#validation-badge");
  if (!status || !badge) return;
  status.textContent = validation?.label || "--";
  badge.childNodes[1].nodeValue = validation?.badge || "Validation --";
  badge.title = validation?.next_actions?.[0] || "Validation terrain requise avant usage product-grade.";
}

function updateRegimeQaUi(regimeQa) {
  const badge = document.querySelector("#qa-badge");
  if (!badge) return;
  badge.classList.toggle("qa-warning", regimeQa?.status === "fix_required" || regimeQa?.status === "review_required");
  badge.classList.toggle("qa-missing", !regimeQa || regimeQa.status === "missing_regime_qa");
  badge.childNodes[1].nodeValue = regimeQa?.badge || "QA priors --";
  badge.title = regimeQa?.title || "QA priors non chargée.";
}

function updateValidationGapUi(validationGaps, overlay, fieldTestPacket) {
  const panel = document.querySelector("#validation-plan");
  const title = document.querySelector("#validation-plan-title");
  const detail = document.querySelector("#validation-plan-detail");
  const fieldTestDetail = document.querySelector("#field-test-plan-detail");
  if (!panel || !title || !detail || !fieldTestDetail) return;
  let panelTitle = fieldTestPacket?.title || "Plan terrain.";
  if (!validationGaps?.gaps?.length) {
    if (!fieldTestPacket?.opportunities?.length) {
      panel.hidden = true;
      return;
    }
    title.textContent = "Plan terrain";
    detail.textContent = "Cas manquants non chargés";
  } else {
    const gap = pickValidationGap(validationGaps, overlay);
    if (!gap) {
      panel.hidden = true;
      return;
    }
    const missing = validationGaps.missingDecisionCases || 0;
    const needed = Number(gap.needed_cases || 0);
    const spot = spotShortLabel(gap.spot_id);
    const regime = regimeShortLabel(gap.regime_id);
    const forecast = gap.current_forecast;
    const forecastText = forecast ? `fenêtre H+${forecast.lead_hour}` : "attendre ce régime";
    title.textContent = `À valider ${missing} cas`;
    detail.textContent = `${spot} ${regime} x${needed} · ${forecastText}`;
    panelTitle = `${gap.why || "Observation terrain requise."} Règle: ${validationGaps.next_field_day_rule || "collecter avant calibration."}`;
  }
  const opportunity = pickFieldTestOpportunity(fieldTestPacket, overlay);
  if (opportunity) {
    const spot = spotShortLabel(opportunity.spot_id);
    const regime = regimeShortLabel(opportunity.regime_id);
    const status = fieldTestStatusLabel(opportunity.status);
    const needed = Number(opportunity.needed_cases || 0);
    fieldTestDetail.textContent = `Test terrain: ${status} · ${spot} ${regime} x${needed}`;
    panel.classList.toggle("validation-plan-stale", opportunity.status === "loaded_forecast_match_past");
    panel.classList.toggle("validation-plan-fresh", opportunity.status === "available_in_loaded_forecast");
    panelTitle = `${panelTitle} ${fieldTestPacket?.title || ""}`.trim();
  } else {
    fieldTestDetail.textContent = "Test terrain: paquet absent";
    panel.classList.remove("validation-plan-stale", "validation-plan-fresh");
  }
  panel.title = panelTitle;
  panel.hidden = false;
}

function updateScaleLabels(maxKnots) {
  document.querySelector("#scale-mid").textContent = `${Math.round(maxKnots / 2)} kt`;
  document.querySelector("#scale-max-label").textContent = `${Math.round(maxKnots)} kt`;
  document.querySelector("#scale-max-value").textContent = `${Math.round(maxKnots)} kt`;
}

function bindScaleControl(overlay) {
  const scaleInput = document.querySelector("#wind-scale");
  scaleInput.value = String(overlay.scaleMaxKnots);
  updateScaleLabels(overlay.scaleMaxKnots);
  scaleInput.addEventListener("input", () => {
    const maxKnots = Number(scaleInput.value);
    overlay.setScaleMaxKnots(maxKnots);
    updateScaleLabels(maxKnots);
  });
}

function buildSessionLegend() {
  const legend = document.querySelector("#session-legend");
  if (!legend || legend.childElementCount) return;
  for (const classId of SESSION_LEGEND_ORDER) {
    const item = SESSION_CLASSES[classId];
    if (!item) continue;
    const chip = document.createElement("span");
    chip.className = "session-chip";
    const swatch = document.createElement("i");
    swatch.className = "session-swatch";
    swatch.style.background = `rgb(${item.rgb.join(", ")})`;
    chip.append(swatch, item.label);
    legend.appendChild(chip);
  }
}

function buildSurfaceLegend() {
  const legend = document.querySelector("#surface-legend");
  if (!legend || legend.childElementCount) return;
  for (const classId of [1, 2, 3, 4, 5, 0]) {
    const item = SURFACE_CLASSES[classId];
    if (!item) continue;
    const chip = document.createElement("span");
    chip.className = "surface-chip";
    const swatch = document.createElement("i");
    swatch.className = "surface-swatch";
    swatch.style.background = `rgb(${item.rgb.join(", ")})`;
    chip.append(swatch, item.label);
    legend.appendChild(chip);
  }
}

function syncModeControls(overlay) {
  const tabs = [...document.querySelectorAll(".mode-tab")];
  const windNinjaAvailable = windNinjaModesAvailable(overlay);
  if (!windNinjaAvailable && overlay.displayMode !== "speed") overlay.displayMode = "speed";
  tabs.forEach((tab) => {
    const mode = tab.dataset.mode || "speed";
    const disabled = !windNinjaAvailable;
    tab.disabled = disabled;
    tab.classList.toggle("disabled", disabled);
    tab.classList.toggle("active", mode === overlay.displayMode);
    tab.setAttribute("aria-disabled", String(disabled));
    tab.title = disabled
      ? "Les modèles bruts s'affichent uniquement en vitesse. Dévente et accélération sont disponibles avec WindNinja."
      : "Mode WindNinja";
  });
}

function bindModeControl(overlay) {
  const tabs = [...document.querySelectorAll(".mode-tab")];
  syncModeControls(overlay);
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      if (tab.disabled) return;
      const mode = tab.dataset.mode || "speed";
      overlay.setDisplayMode(mode);
    });
  });
}

function syncLayerControls(overlay) {
  const buttons = [...document.querySelectorAll(".layer-toggle, .map-layer-button")];
  for (const button of buttons) {
    const layer = button.dataset.layer;
    if (layer && Object.prototype.hasOwnProperty.call(overlay.visibleLayers, layer)) {
      const model = rawModelForKey(overlay, layer);
      const loading = Boolean(overlay.rawLayerLoading?.[layer]);
      const lazyLoadable = isRawLayerKey(layer) && !model && Boolean(rawModelUrl(layer));
      const equivalentStep = model ? equivalentStepForLayer(overlay, layer) : null;
      const unavailable = isRawLayerKey(layer) && !loading && !lazyLoadable && (!model || !equivalentStep);
      const visible = Boolean(overlay.visibleLayers[layer]);
      button.classList.toggle("active", visible);
      button.classList.toggle("disabled", unavailable);
      button.classList.toggle("loading", loading);
      button.disabled = unavailable || loading;
      button.setAttribute("aria-pressed", String(visible));
      button.setAttribute("aria-disabled", String(unavailable || loading));
      if (loading) {
        button.title = `Chargement ${rawLayerLabel(layer)}...`;
      } else if (lazyLoadable) {
        const label = rawLayerLabel(layer);
        button.title = overlay.rawLayerLoadError?.[layer]
          ? `${label} non chargé: cliquez pour réessayer`
          : `Charger ${label}`;
      } else if (unavailable) {
        const label = rawLayerLabel(layer);
        button.title = model
          ? `${label} hors tranche ${formatForecastDay(displayedValidTimeUtc(overlay))} ${formatClock(displayedValidTimeUtc(overlay))}`
          : `${label} indisponible: générez ${layer === "icon2i" ? "icon2i" : layer}-corsica-latest.json`;
      } else if (isRawLayerKey(layer)) {
        const label = rawLayerLabel(layer);
        button.title = `${label} H+${equivalentStep.lead_hour} disponible pour ${formatForecastDay(equivalentStep.valid_time_utc)} ${formatClock(equivalentStep.valid_time_utc)}`;
      }
    }
  }
}

function bindLayerControl(overlay, payload) {
  const buttons = [...document.querySelectorAll(".layer-toggle, .map-layer-button")];
  syncLayerControls(overlay);
  buttons.forEach((button) => {
    button.addEventListener("click", async () => {
      const layer = button.dataset.layer;
      if (button.disabled || !layer) return;
      const nextVisible = !button.classList.contains("active");
      if (nextVisible && isRawLayerKey(layer) && !rawModelForKey(overlay, layer)) {
        const loaded = await ensureRawModelLoaded(overlay, layer);
        if (!loaded) {
          syncLayerControls(overlay);
          updateReadout(payload, overlay);
          return;
        }
      }
      overlay.setLayerVisible(layer, nextVisible);
      syncLayerControls(overlay);
      buildForecastButtons(payload, overlay);
      updateReadout(payload, overlay);
    });
  });
}

function bindParticleControl(overlay) {
  const button = document.querySelector("#particle-toggle");
  if (!button) return;
  button.classList.toggle("active", overlay.particlesEnabled);
  button.setAttribute("aria-pressed", String(overlay.particlesEnabled));
  button.addEventListener("click", () => {
    const nextEnabled = !button.classList.contains("active");
    button.classList.toggle("active", nextEnabled);
    button.setAttribute("aria-pressed", String(nextEnabled));
    overlay.setParticlesEnabled(nextEnabled);
  });
}

function bindParticleSliders(overlay) {
  const opacityInput = document.querySelector("#particle-opacity");
  const opacityValue = document.querySelector("#particle-opacity-value");
  const densityInput = document.querySelector("#particle-density");
  const densityValue = document.querySelector("#particle-density-value");
  const lifeInput = document.querySelector("#particle-life");
  const lifeValue = document.querySelector("#particle-life-value");
  const sizeInput = document.querySelector("#particle-size");
  const sizeValue = document.querySelector("#particle-size-value");
  if (opacityInput && opacityValue) {
    opacityInput.value = String(Math.round(overlay.particleOpacity * 100));
    opacityValue.textContent = `${opacityInput.value}%`;
    opacityInput.addEventListener("input", () => {
      const value = Number(opacityInput.value);
      opacityValue.textContent = `${value}%`;
      overlay.setParticleOpacity(value / 100);
    });
  }
  if (densityInput && densityValue) {
    densityInput.value = String(Math.round(overlay.particleDensity * 100));
    densityValue.textContent = `${densityInput.value}%`;
    densityInput.addEventListener("input", () => {
      const value = Number(densityInput.value);
      densityValue.textContent = `${value}%`;
      overlay.setParticleDensity(value / 100);
    });
  }
  if (lifeInput && lifeValue) {
    lifeInput.value = String(Math.round(overlay.particleLifeScale * 100));
    lifeValue.textContent = `${lifeInput.value}%`;
    lifeInput.addEventListener("input", () => {
      const value = Number(lifeInput.value);
      lifeValue.textContent = `${value}%`;
      overlay.setParticleLifeScale(value / 100);
    });
  }
  if (sizeInput && sizeValue) {
    sizeInput.value = String(Math.round(overlay.particleSizeScale * 100));
    sizeValue.textContent = `${sizeInput.value}%`;
    sizeInput.addEventListener("input", () => {
      const value = Number(sizeInput.value);
      sizeValue.textContent = `${value}%`;
      overlay.setParticleSizeScale(value / 100);
    });
  }
}

function bindMapFocusControl() {
  const shell = document.querySelector(".app-shell");
  const button = document.querySelector("#map-focus-toggle");
  if (!shell || !button) return;

  const setHidden = (hidden) => {
    shell.classList.toggle("ui-hidden", hidden);
    button.setAttribute("aria-pressed", String(hidden));
    button.textContent = hidden ? "Menus" : "Carte";
    button.title = hidden ? "Afficher les menus" : "Masquer les menus";
  };

  button.addEventListener("click", () => {
    setHidden(!shell.classList.contains("ui-hidden"));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && shell.classList.contains("ui-hidden")) {
      setHidden(false);
    }
  });
}

function bindLegendCompactControl() {
  const panel = document.querySelector(".legend-panel");
  const button = document.querySelector("#legend-expand-toggle");
  const shell = document.querySelector(".app-shell");
  if (!panel || !button) return;

  const setCollapsed = (collapsed) => {
    panel.classList.toggle("mobile-collapsed", collapsed);
    shell?.classList.toggle("legend-controls-open", !collapsed);
    button.setAttribute("aria-expanded", String(!collapsed));
    button.textContent = collapsed ? "Réglages" : "Fermer";
  };

  setCollapsed(true);
  button.addEventListener("click", () => {
    setCollapsed(!panel.classList.contains("mobile-collapsed"));
  });
}

function updateLegendTitle(mode) {
  const title = document.querySelector("#legend-title");
  const continuousLegend = document.querySelector(".gradient-bar");
  const scaleRow = document.querySelector(".legend-row");
  const scaleControl = document.querySelector(".scale-control");
  const sessionLegend = document.querySelector("#session-legend");
  const surfaceLegend = document.querySelector("#surface-legend");
  if (continuousLegend) continuousLegend.hidden = false;
  if (scaleRow) scaleRow.hidden = false;
  if (scaleControl) scaleControl.hidden = false;
  if (sessionLegend) sessionLegend.hidden = true;
  if (surfaceLegend) surfaceLegend.hidden = true;
  const labels = {
    speed: "Vitesse du vent",
    devente: "Dévente WindNinja",
    acceleration: "Accélération WindNinja",
  };
  title.textContent = labels[mode] || labels.speed;
}

function bindCfdControl(map, overlay) {
  const button = document.querySelector("#cfd-focus");
  const coastalButton = document.querySelector("#coastal-cfd-focus");
  const bayButton = document.querySelector("#bay-model-focus");
  const status = document.querySelector("#cfd-status");
  if (!overlay.cfd && !overlay.coastalTiles && !overlay.bayModel && !overlay.multiscalePlan && !overlay.expandedWind && !overlay.localWind && !overlay.windNinjaCorsicaTiles) {
    status.classList.add("cfd-disabled");
    button.disabled = true;
    coastalButton.disabled = true;
    bayButton.disabled = true;
    return;
  }
  status.classList.remove("cfd-disabled");
  const parts = [];
  if (overlay.cfd) parts.push(`${overlay.cfd.samples.length} points 0.5 m`);
  if (overlay.coastalTiles) parts.push(`${overlay.coastalTiles.tileCount} tuiles planifiees`);
  if (overlay.bayModel) {
    const count = overlay.bayModel.summary.selected_tile_count || overlay.bayModel.tiles.length;
    parts.push(`plan 1 m ${count} tuiles`);
  }
  if (overlay.multiscalePlan?.targetBounds) parts.push("cible 5 km");
  if (overlay.expandedWind) {
    const resolution = overlay.expandedWind.domain?.grid_resolution_m || "?";
    parts.push(`overview 5 km ${resolution} m`);
  }
  if (overlay.rasterTiles) {
    parts.push(`raster ${overlay.rasterTiles.tileCount || "?"} tuiles`);
  }
  if (overlay.windNinjaCorsicaTiles) {
    const resolution = overlay.windNinjaCorsicaTiles.source?.resolution_m || "?";
    parts.push(`WindNinja Corse ${resolution} m`);
  }
  status.childNodes[1].nodeValue = `CFD ${parts.join(" + ")}`;
  if (overlay.cfd) {
    button.disabled = false;
    button.title = `${overlay.cfd.label} · ${overlay.cfd.samples.length} points`;
    button.addEventListener("click", () => {
      map.fitBounds(
        [
          [overlay.cfd.bounds.minLat, overlay.cfd.bounds.minLon],
          [overlay.cfd.bounds.maxLat, overlay.cfd.bounds.maxLon],
        ],
        { padding: [120, 120], maxZoom: 16 }
      );
    });
  }
  if (overlay.coastalTiles) {
    coastalButton.disabled = false;
    coastalButton.title = `${overlay.coastalTiles.tileCount} tuiles CFD cotieres planifiees`;
    coastalButton.addEventListener("click", () => {
      map.fitBounds(
        [
          [overlay.coastalTiles.bounds.minLat, overlay.coastalTiles.bounds.minLon],
          [overlay.coastalTiles.bounds.maxLat, overlay.coastalTiles.bounds.maxLon],
        ],
        { padding: [80, 80], maxZoom: 12 }
      );
      overlay.heatDirty = true;
      overlay.draw();
    });
  }
  if (overlay.bayModel) {
    bayButton.disabled = false;
    bayButton.title = `Plan baie 1 m · ${overlay.bayModel.tiles.length} tuiles · ${overlay.bayModel.summary.openfoam_micro50_candidate || 0} poches 0.5 m`;
    bayButton.addEventListener("click", () => {
      map.fitBounds(
        [
          [overlay.bayModel.bounds.minLat, overlay.bayModel.bounds.minLon],
          [overlay.bayModel.bounds.maxLat, overlay.bayModel.bounds.maxLon],
        ],
        { padding: [84, 84], maxZoom: 12 }
      );
      overlay.heatDirty = true;
      overlay.draw();
    });
  }
  if (overlay.localWind) {
    const resolution = overlay.localWind.domain?.grid_resolution_m || "?";
    parts.push(`grille 2 m ${resolution} m`);
    if (overlay.spotGrids?.spots?.length) {
      const visibleSpots = overlay.spotGrids.spots.filter((spot) => spot.ui_visible !== false);
      const focus5m = overlay.spotGrids.spots.filter((spot) => spot.ui_visible === false && spot.resolution_m <= 5).length;
      parts.push(`${visibleSpots.length} spots 10-20 m${focus5m ? ` + ${focus5m} poches 5 m` : ""}`);
    }
    if (overlay.windNinjaSpots?.spots?.length) {
      parts.push(`${overlay.windNinjaSpots.spots.length} WindNinja 5 m`);
    }
    status.childNodes[1].nodeValue = `Local ${parts.join(" + ")}`;
  }
  refreshCoverageStatus(overlay);
}

function refreshCoverageStatus(overlay) {
  const status = document.querySelector("#cfd-status");
  if (!status) return;
  const parts = [];
  parts.push(`AROME ${overlay.visibleLayers.arome ? "ON" : "OFF"}`);
  parts.push(overlay.aromepi ? `AROME-PI ${overlay.visibleLayers.aromepi ? (overlay.aromePiStep ? "ON" : "hors échéance") : "OFF"}` : "AROME-PI indisponible");
  parts.push(overlay.moloch ? `MOLOCH ${overlay.visibleLayers.moloch ? (overlay.molochStep ? "ON" : "hors échéance") : "OFF"}` : "MOLOCH indisponible");
  parts.push(overlay.icon2i ? `ICON-2I ${overlay.visibleLayers.icon2i ? (overlay.icon2iStep ? "ON" : "hors échéance") : "OFF"}` : "ICON-2I indisponible");
  parts.push(windNinjaStatusLabel(overlay, overlay.windNinjaCorsica50mTiles, "windninja50", "WN 50 m"));
  status.childNodes[1].nodeValue = parts.join(" + ");
}

function windNinjaStatusLabel(overlay, tileState, layerKey, label) {
  if (!tileState) return `${label} indisponible`;
  const resolution = tileState.source?.resolution_m || "?";
  const count = tileState.source?.tile_count || tileState.tileCount || "?";
  const stepKey = windNinjaCorsicaStepKey(overlay, tileState);
  const state = overlay.visibleLayers[layerKey] ? (stepKey ? "ON" : "hors échéance") : "OFF";
  return `${label} ${state} (${resolution} m horiz, ${count})`;
}

function buildSpotButtons(map, overlay) {
  const strip = document.querySelector(".spot-strip");
  if (!strip) return;
  strip.innerHTML = "";
  const spots = [
    ...(overlay.spotGrids?.spots?.filter((item) => item.ui_visible !== false) || []),
    ...(overlay.windNinjaSpots?.spots || []),
  ];
  if (!spots.length) {
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  for (const spot of spots) {
    const button = document.createElement("button");
    button.className = "spot-button";
    button.type = "button";
    button.dataset.spotId = spot.id;
    const isWindNinja = spot.source === "WindNinja diagnostic solve";
    button.textContent = isWindNinja ? `WN ${spot.label.split(" ")[0]}` : spot.label.split(" / ")[0];
    button.title = `${spot.label} · ${isWindNinja ? "WindNinja" : "grille"} ${spot.resolution_m} m`;
    button.addEventListener("click", () => {
      for (const child of strip.children) child.classList.remove("active");
      button.classList.add("active");
      map.fitBounds(
        [
          [spot.bounds.minLat, spot.bounds.minLon],
          [spot.bounds.maxLat, spot.bounds.maxLon],
        ],
        { paddingTopLeft: [24, 120], paddingBottomRight: [24, 120], maxZoom: spot.resolution_m <= 10 ? 15 : 14 }
      );
      overlay.heatDirty = true;
      overlay.draw();
      window.setTimeout(() => {
        updatePointReadout(L.latLng(spot.center.lat, spot.center.lon), overlay);
      }, 320);
    });
    strip.appendChild(button);
  }
}

function buildForecastButtons(payload, overlay) {
  const strip = document.querySelector(".forecast-strip");
  strip.innerHTML = "";
  const model = activeRawModel(overlay) || payload;
  const modelKey = rawModelKey(overlay);
  const modelShortLabel = rawLayerLabel(modelKey);
  const steps = model.forecast_steps || [];
  strip.dataset.model = modelKey;
  const activeStep = forecastStepByLead(model, overlay.activeLeadHour) || steps[0];
  const days = forecastDayGroups(steps);
  const activeDayKey = activeStep?.valid_time_utc ? localDayKey(activeStep.valid_time_utc) : days[0]?.key;
  const activeDayIndex = Math.max(0, days.findIndex((day) => day.key === activeDayKey));
  const activeDay = days[activeDayIndex] || days[0] || { steps };
  const daySteps = activeDay.steps || steps;
  const selectStep = (step) => {
    if (!step) return;
    const leadHour = Number(step.lead_hour);
    const aromeIndex = forecastIndexByLead(payload, leadHour);
    overlay.setStep(aromeIndex >= 0 ? aromeIndex : overlay.stepIndex, leadHour);
    buildForecastButtons(payload, overlay);
    syncLayerControls(overlay);
    updateReadout(payload, overlay);
  };
  const buildDayNav = (day, direction) => {
    const button = document.createElement("button");
    button.className = `forecast-day-nav forecast-day-nav-${direction}`;
    button.type = "button";
    const isDisabled = !day?.steps?.length;
    button.disabled = isDisabled;
    button.setAttribute("aria-disabled", String(isDisabled));
    const dateParts = day?.steps?.[0] ? formatDayNavParts(day.steps[0].valid_time_utc) : { day: "--", month: "" };
    const dayNumber = document.createElement("strong");
    dayNumber.textContent = dateParts.day;
    const month = document.createElement("span");
    month.textContent = dateParts.month;
    button.append(dayNumber, month);
    if (day?.lastMs < Date.now() - 15 * 60 * 1000) button.classList.add("past");
    button.title = day?.steps?.length
      ? `${direction === "prev" ? "Jour précédent" : "Jour suivant"} · ${formatForecastDay(day.steps[0].valid_time_utc)}`
      : direction === "prev"
        ? "Aucun jour précédent disponible"
        : "Aucun jour suivant disponible";
    button.setAttribute("aria-label", button.title);
    button.addEventListener("click", () => {
      if (button.disabled) return;
      selectStep(closestStepInDay(day, activeStep));
    });
    return button;
  };
  const shell = document.createElement("div");
  shell.className = "forecast-timeline-shell";
  strip.appendChild(shell);
  shell.appendChild(buildDayNav(days[activeDayIndex - 1], "prev"));
  const main = document.createElement("div");
  main.className = "forecast-timeline-main";
  shell.appendChild(main);
  if (activeStep) {
    const detail = document.createElement("div");
    detail.className = "forecast-selection";
    detail.setAttribute("aria-live", "polite");
    const modelLabel = document.createElement("span");
    modelLabel.className = "forecast-selection-model";
    modelLabel.textContent = modelShortLabel;
    const valid = document.createElement("strong");
    valid.textContent = `${formatForecastDay(activeStep.valid_time_utc)} ${formatClock(activeStep.valid_time_utc)}`;
    const lead = document.createElement("span");
    lead.textContent = formatLeadLabel(activeStep);
    const run = document.createElement("span");
    run.textContent = `Run ${formatRunStamp(model.run_time_utc)}`;
    detail.append(modelLabel, valid, lead, run);
    main.appendChild(detail);
  }
  const track = document.createElement("div");
  track.className = "forecast-track";
  track.setAttribute("role", "listbox");
  track.setAttribute("aria-label", `Échéances ${modelShortLabel} ${activeDay?.steps?.[0] ? formatForecastDay(activeDay.steps[0].valid_time_utc) : ""}`);
  main.appendChild(track);
  shell.appendChild(buildDayNav(days[activeDayIndex + 1], "next"));
  let activeButton = null;
  daySteps.forEach((step, index) => {
    const leadHour = Number(step.lead_hour);
    const hasWindNinja = hasWindNinja50Step(overlay, leadHour);
    const isPast = new Date(step.valid_time_utc).getTime() < Date.now() - 15 * 60 * 1000;
    const button = document.createElement("button");
    button.className = `forecast-step${leadHour === Number(overlay.activeLeadHour) ? " active" : ""}${hasWindNinja ? " windninja-ready" : ""}${isPast ? " past" : ""}${index % 2 ? " label-bottom" : " label-top"}`;
    button.type = "button";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(leadHour === Number(overlay.activeLeadHour)));
    if (leadHour === Number(overlay.activeLeadHour)) activeButton = button;
    button.dataset.leadHour = String(leadHour);
    button.dataset.model = modelKey;
    const top = document.createElement("span");
    top.className = "forecast-step-label forecast-step-label-top";
    const marker = document.createElement("span");
    marker.className = "forecast-step-marker";
    const bottom = document.createElement("span");
    bottom.className = "forecast-step-label forecast-step-label-bottom";
    const hourText = formatClock(step.valid_time_utc);
    if (index % 2) bottom.textContent = hourText;
    else top.textContent = hourText;
    button.append(top, marker, bottom);
    button.title =
      `Prévision ${formatForecastDay(step.valid_time_utc)} ${formatClock(step.valid_time_utc)} · ` +
      `run ${modelShortLabel} ${formatRunStamp(model.run_time_utc)} · ` +
      `${hasWindNinja ? "WindNinja 50 m disponible pour " + formatLeadLabel(step) : `${model.model_label} brut`}`;
    button.setAttribute(
      "aria-label",
      `Prévision ${formatForecastDay(step.valid_time_utc)} ${formatClock(step.valid_time_utc)}, ` +
        `calculée par le run ${modelShortLabel} du ${formatRunStamp(model.run_time_utc)}`
    );
    button.addEventListener("click", () => {
      selectStep(step);
    });
    track.appendChild(button);
  });
  enableTimelineDragScroll(track);
  if (activeButton) {
    requestAnimationFrame(() => {
      activeButton.scrollIntoView({ block: "nearest", inline: "center" });
    });
  }
}

// Let the forecast timeline be dragged left/right with the mouse (it overflows when a model has
// many échéances, e.g. MOLOCH). Touch keeps Leaflet/native horizontal scroll (touch-action: pan-x),
// so we only hijack mouse pointers here. A drag past a few px suppresses the click that would
// otherwise select a step.
function enableTimelineDragScroll(track) {
  let dragging = false;
  let startX = 0;
  let startScroll = 0;
  let moved = false;
  let pointerId = null;

  track.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "mouse" || event.button !== 0) return;
    dragging = true;
    moved = false;
    track._suppressClick = false;
    startX = event.clientX;
    startScroll = track.scrollLeft;
    pointerId = event.pointerId;
    track.classList.add("dragging");
  });

  track.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - startX;
    if (!moved && Math.abs(dx) > 4) {
      moved = true;
      track.setPointerCapture?.(pointerId);
    }
    if (moved) {
      track.scrollLeft = startScroll - dx;
      event.preventDefault();
    }
  });

  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    track.classList.remove("dragging");
    try {
      if (pointerId !== null) track.releasePointerCapture?.(pointerId);
    } catch {
      /* pointer already released */
    }
    if (moved) {
      // The click that fires right after this pointerup is swallowed below; auto-clear on the next
      // task so a drag that produces no click can never eat a later legitimate selection click.
      track._suppressClick = true;
      window.setTimeout(() => {
        track._suppressClick = false;
      }, 0);
    }
  };
  track.addEventListener("pointerup", endDrag);
  track.addEventListener("pointercancel", endDrag);

  // A drag ends with a click event on the step under the cursor — swallow it so dragging doesn't
  // also change the selected échéance.
  track.addEventListener(
    "click",
    (event) => {
      if (track._suppressClick) {
        event.stopPropagation();
        event.preventDefault();
        track._suppressClick = false;
      }
    },
    true
  );

  // Mouse wheel over the timeline scrolls it horizontally (the strip is outside the map, so this
  // never triggers map zoom).
  track.addEventListener(
    "wheel",
    (event) => {
      const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
      if (!delta) return;
      track.scrollLeft += delta;
      event.preventDefault();
    },
    { passive: false }
  );
}

function bindMapReadout(map, overlay) {
  let inspectorLocked = false;
  let lockedContainerPoint = null;
  let selectedPointMarker = null;
  let pendingFrame = null;
  let touchMoved = false;

  const ensureSelectedPointMarker = (containerPoint) => {
    if (!containerPoint) return;
    if (!selectedPointMarker) {
      selectedPointMarker = document.createElement("div");
      selectedPointMarker.className = "point-inspector-anchor";
      selectedPointMarker.setAttribute("aria-hidden", "true");
      document.querySelector(".app-shell").appendChild(selectedPointMarker);
    }
    selectedPointMarker.style.transform = `translate(${Math.round(containerPoint.x - 5)}px, ${Math.round(containerPoint.y - 5)}px)`;
  };

  const clearSelectedPointMarker = () => {
    if (!selectedPointMarker) return;
    selectedPointMarker.remove();
    selectedPointMarker = null;
  };

  const scheduleInspect = (event, locked = inspectorLocked) => {
    if (inspectorLocked && !locked) return;
    if (pendingFrame) cancelAnimationFrame(pendingFrame);
    pendingFrame = requestAnimationFrame(() => {
      pendingFrame = null;
      const containerPoint = event.containerPoint || map.latLngToContainerPoint(event.latlng);
      updatePointInspector(event.latlng, containerPoint, overlay, map, locked);
    });
  };

  const refreshLockedInspection = () => {
    if (!inspectorLocked || !lockedContainerPoint) return;
    const size = map.getSize();
    if (
      lockedContainerPoint.x < 0 ||
      lockedContainerPoint.y < 0 ||
      lockedContainerPoint.x > size.x ||
      lockedContainerPoint.y > size.y
    ) {
      hidePointInspector();
      return;
    }
    const latlng = map.containerPointToLatLng(lockedContainerPoint);
    ensureSelectedPointMarker(lockedContainerPoint);
    scheduleInspect({ latlng, containerPoint: lockedContainerPoint }, true);
    updatePointReadout(latlng, overlay);
  };
  overlay.refreshPinnedPointInspector = refreshLockedInspection;

  const touchEventToInspectEvent = (touch) => {
    const rect = map.getContainer().getBoundingClientRect();
    const containerPoint = L.point(touch.clientX - rect.left, touch.clientY - rect.top);
    return {
      latlng: map.containerPointToLatLng(containerPoint),
      containerPoint,
    };
  };

  const inspectTouch = (touch) => {
    const inspectEvent = touchEventToInspectEvent(touch);
    scheduleInspect(inspectEvent, false);
    updatePointReadout(inspectEvent.latlng, overlay);
  };

  map.on("mousemove click", (event) => {
    if (event.type === "click") {
      if (touchMoved) {
        touchMoved = false;
        return;
      }
      inspectorLocked = !inspectorLocked;
      if (!inspectorLocked) {
        lockedContainerPoint = null;
        clearSelectedPointMarker();
        hidePointInspector();
        return;
      }
      lockedContainerPoint = event.containerPoint;
      ensureSelectedPointMarker(lockedContainerPoint);
      refreshLockedInspection();
      return;
    }
    if (!inspectorLocked) {
      scheduleInspect(event, false);
    }
    updatePointReadout(event.latlng, overlay);
  });
  map.on("mouseout", () => {
    if (!inspectorLocked) hidePointInspector();
  });
  map.on("move zoom resize", refreshLockedInspection);
  map.getContainer().addEventListener(
    "touchstart",
    (event) => {
      if (!event.touches?.length) return;
      touchMoved = false;
      inspectTouch(event.touches[0]);
    },
    { passive: true }
  );
  map.getContainer().addEventListener(
    "touchmove",
    (event) => {
      if (!event.touches?.length) return;
      touchMoved = true;
      if (!inspectorLocked) inspectTouch(event.touches[0]);
    },
    { passive: true }
  );
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && inspectorLocked) {
      inspectorLocked = false;
      lockedContainerPoint = null;
      clearSelectedPointMarker();
      hidePointInspector();
    }
  });
}

function hidePointInspector() {
  const inspector = document.querySelector("#point-inspector");
  if (!inspector) return;
  inspector.hidden = true;
  inspector.classList.remove("locked");
}

function updatePointInspector(latlng, containerPoint, overlay, map, locked = false) {
  const inspector = document.querySelector("#point-inspector");
  if (!inspector || !latlng || !containerPoint) return;
  const point = buildPointInspection(latlng, overlay);
  if (!point) {
    inspector.hidden = true;
    inspector.classList.remove("locked");
    return;
  }
  inspector.hidden = false;
  inspector.classList.toggle("locked", locked);
  inspector.classList.toggle("point-inspector--metrics", Boolean(point.metricsHtml));
  const speedNode = document.querySelector("#point-inspector-speed");
  const detailNode = document.querySelector("#point-inspector-detail");
  if (point.metricsHtml) {
    speedNode.innerHTML = point.metricsHtml;
    detailNode.textContent = "";
  } else {
    speedNode.textContent = point.speedText;
    detailNode.textContent = point.detail;
  }
  inspector.title = [locked ? "Point épinglé" : null, point.source, point.title || point.detail].filter(Boolean).join(" · ");
  positionPointInspector(inspector, containerPoint, map);
  if (point.loading) {
    const retryCount = Number(inspector.dataset.loadingRetry || 0);
    if (retryCount >= 8) return;
    inspector.dataset.loadingRetry = String(retryCount + 1);
    window.setTimeout(() => {
      updatePointInspector(latlng, containerPoint, overlay, map, locked);
    }, 260);
  } else {
    inspector.dataset.loadingRetry = "0";
  }
}

function positionPointInspector(inspector, containerPoint, map) {
  const size = map.getSize();
  const width = inspector.offsetWidth || 92;
  const height = inspector.offsetHeight || 56;
  let x = containerPoint.x + 14;
  let y = containerPoint.y + 14;
  if (x + width > size.x - 8) x = containerPoint.x - width - 14;
  if (y + height > size.y - 8) y = containerPoint.y - height - 14;
  x = Math.max(8, Math.min(size.x - width - 8, x));
  y = Math.max(8, Math.min(size.y - height - 8, y));
  inspector.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
}

function buildPointInspection(latlng, overlay) {
  const aromeField = overlay.aromeFieldAt(latlng, true);
  const rawField = overlay.visibleLayers.aromepi
    ? overlay.aromePiFieldAt(latlng, true)
    : overlay.visibleLayers.icon2i
    ? overlay.icon2iFieldAt(latlng, true)
    : overlay.visibleLayers.moloch
      ? overlay.molochFieldAt(latlng, true)
      : aromeField;
  const windNinja = windNinjaPointInspection(latlng, overlay, aromeField);
  if (windNinja) return windNinja;
  if (!rawField || !anyRawLayerVisible(overlay)) return null;
  const label = rawLayerLabel(rawModelKey(overlay));
  const source = `${label} 10 m`;
  const note = rawField.gustSpeedKnots && rawField.meanSpeedKnots
    ? `rafale ${rawField.gustSpeedKnots.toFixed(1)} kt · moyen ${rawField.meanSpeedKnots.toFixed(1)} kt`
    : `${label} brut`;
  const meanKnots = rawField.meanSpeedKnots ?? rawField.speedKnots;
  return {
    source,
    speedText: `${meanKnots.toFixed(1)} kt`,
    metricsHtml: compactWindMetricsHtml({
      meanKnots,
      gustKnots: rawField.gustSpeedKnots,
      windFromDeg: rawField.windFromDeg,
    }),
    detail: compactPointDetail({ windFromDeg: rawField.windFromDeg }),
    title: fullPointDetail({
      windFromDeg: rawField.windFromDeg,
      heightLabel: rawField.heightLabel,
      resolutionLabel: rawField.resolutionLabel,
      latlng,
      note,
    }),
  };
}

function windNinjaPointInspection(latlng, overlay, aromeField) {
  const layers = [
    { key: "windninja50", label: "WindNinja", tileState: overlay.windNinjaCorsica50mTiles },
  ];
  let hasCandidate = false;
  for (const layer of layers) {
    const tileState = layer.tileState;
    if (!overlay.visibleLayers[layer.key] || !tileState || tileState.encoding !== "data") continue;
    if (!windNinjaCorsicaStepKey(overlay, tileState)) continue;
    if (tileState.bounds && !tileState.bounds.contains(latlng)) continue;
    hasCandidate = true;
    const sample = overlay.windNinjaDataSampleAt(latlng, tileState, layer.key);
    if (!sample) continue;
    const resolution = tileState.source?.resolution_m || "?";
    const height = tileState.source?.output_height_m || "?";
    const ratioPct = Math.round((sample.ratio - 1) * 100);
    const ratioLabel = ratioPct >= 0 ? `accél. +${ratioPct}%` : `dévente ${ratioPct}%`;
    const coverage = Math.round(sample.coverage * 100);
    return {
      source: `${layer.label} ${resolution} m / ${height} m`,
      speedText: `${sample.speedKt.toFixed(1)} kt`,
      metricsHtml: compactWindMetricsHtml({
        meanKnots: sample.speedKt,
        windFromDeg: aromeField?.windFromDeg,
      }),
      detail: compactPointDetail({ windFromDeg: aromeField?.windFromDeg }),
      title: fullPointDetail({
        windFromDeg: aromeField?.windFromDeg,
        heightLabel: `${height} m AGL`,
        resolutionLabel: `${resolution} m`,
        latlng,
        note: `${ratioLabel} · couverture ${coverage}%`,
      }),
    };
  }
  if (hasCandidate && aromeField) {
    return {
      source: "WindNinja data",
      speedText: `${aromeField.speedKnots.toFixed(1)} kt`,
      loading: true,
      metricsHtml: compactWindMetricsHtml({
        meanKnots: aromeField.speedKnots,
        windFromDeg: aromeField.windFromDeg,
      }),
      detail: compactPointDetail({ windFromDeg: aromeField.windFromDeg }),
      title: fullPointDetail({
        windFromDeg: aromeField.windFromDeg,
        heightLabel: aromeField.heightLabel,
        resolutionLabel: aromeField.resolutionLabel,
        latlng,
        note: "tuile en chargement · AROME temporaire",
      }),
    };
  }
  return null;
}

function compactWindMetricsHtml({ meanKnots, gustKnots, windFromDeg }) {
  const direction = windFromDeg === null || windFromDeg === undefined ? "--" : `${Math.round(windFromDeg)}°`;
  const rows = [
    `<span class="point-inspector__metric"><span>Moy.</span><strong>${meanKnots.toFixed(1)}</strong></span>`,
  ];
  if (Number.isFinite(gustKnots)) {
    rows.push(`<span class="point-inspector__metric point-inspector__metric--gust"><span>Raf.</span><strong>${gustKnots.toFixed(1)}</strong></span>`);
  }
  rows.push(`<span class="point-inspector__metric point-inspector__metric--dir"><span>Dir.</span><strong>${direction}</strong></span>`);
  return rows.join("");
}

function compactPointDetail({ windFromDeg, note }) {
  const windFrom = windFromDeg === null || windFromDeg === undefined ? "--" : `${Math.round(windFromDeg)}°`;
  return note ? `de ${windFrom} · ${note}` : `de ${windFrom}`;
}

function fullPointDetail({ windFromDeg, heightLabel, resolutionLabel, latlng, note }) {
  const windFrom = windFromDeg === null || windFromDeg === undefined ? "--" : `${Math.round(windFromDeg)}°`;
  const coords = `${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}`;
  return `${heightLabel} · ${resolutionLabel} · vent de ${windFrom} · ${note} · ${coords}`;
}

function updatePointReadout(latlng, overlay) {
  const field = overlay.fieldAt(latlng);
  if (!field) {
    document.querySelector(".readout-heading").textContent = rawLayerLabel(rawModelKey(overlay));
    document.querySelector("#spot-speed").textContent = "--";
    const rawVisible = anyRawLayerVisible(overlay);
    const rawName = rawLayerLabel(rawModelKey(overlay));
    document.querySelector("#spot-detail").textContent = rawVisible ? `Hors domaine ${rawName}` : "Couche météo masquée";
    return;
  }
  const speedKnots = field.speedKnots ?? field.speed * KNOTS_PER_MPS;
  document.querySelector(".readout-heading").textContent = field.spot
    ? `${field.spotLabel} · ${field.resolutionLabel}`
    : field.corridor
      ? `${field.corridorLabel} · ${field.resolutionLabel}`
    : field.local2m
      ? `${field.sourceLabel} · ${field.resolutionLabel}`
      : `${field.sourceLabel} · ${field.resolutionLabel}`;
  document.querySelector("#spot-speed").textContent = `${speedKnots.toFixed(0)} kt`;
  if (field.local2m) {
    if (overlay.displayMode === "surface") {
      const surfaceClass = SURFACE_CLASSES[field.surfaceClassId] || SURFACE_CLASSES[0];
      const surfaceConfidence = Math.round((field.surfaceConfidence || 0) * 100);
      const coast = field.distanceToCoastM === null ? "--" : `${Math.round(field.distanceToCoastM)} m`;
      const coastal = Math.round((field.coastalBand || 0) * 100);
      document.querySelector("#spot-detail").textContent =
        `${surfaceClass.label} · côte ${coast} · bande ${coastal}% · conf. surface ${surfaceConfidence}%`;
      return;
    }
    const devente = Math.round((field.devente || 0) * 100);
    const acceleration = Math.round((field.acceleration || 0) * 100);
    const confidence = Math.round((field.confidence || 0) * 100);
    const modelConfidence = Math.round((field.modelConfidence || field.confidence || 0) * 100);
    const quality = Math.round((field.quality || 0) * 100);
    const modelQuality = Math.round((field.modelQuality || field.quality || 0) * 100);
    const validationLabel = overlay.validation?.status === "product_grade_candidate" ? "validé" : "pré-valid.";
    const verdict = SESSION_CLASSES[field.sessionClassId]?.label || "Verdict modèle";
    const layerNote = field.windNinja ? "WindNinja 5 m" : field.corridor ? "corridor prioritaire" : field.overview2m ? "overview baie" : "local spot";
    document.querySelector("#spot-detail").textContent =
      `${field.heightLabel} · ${layerNote} · ${verdict} · session ${quality}% (${validationLabel}) · modèle ${modelQuality}% · dévente ${devente}% · accél. ${acceleration}% · conf. ${confidence}%/${modelConfidence}%`;
  } else {
    const windFrom = field.windFromDeg === null || field.windFromDeg === undefined ? "--" : `${Math.round(field.windFromDeg)}°`;
    const suffix = "champ météo brut";
    document.querySelector("#spot-detail").textContent = `${field.heightLabel} · vent de ${windFrom} · ${suffix}`;
  }
}

// Continuous wheel/trackpad zoom that glides toward the cursor, instead of Leaflet's stepped
// debounced wheel zoom. It eases the live zoom toward an accumulated target every frame using
// map._move (transform only, no tile reload) and settles via _moveEnd — much closer to the
// "glued to the gesture" feel of Google Maps. Touch pinch is already continuous in Leaflet, so
// this only replaces the desktop wheel/trackpad path. Adapted from the Leaflet.SmoothWheelZoom
// pattern (Mutsuyuki, MIT).
const SmoothWheelZoom = L.Handler.extend({
  addHooks() {
    L.DomEvent.on(this._map._container, "wheel", this._onWheelScroll, this);
  },
  removeHooks() {
    L.DomEvent.off(this._map._container, "wheel", this._onWheelScroll, this);
  },
  _onWheelScroll(e) {
    if (!this._isWheeling) this._onWheelStart(e);
    this._onWheeling(e);
  },
  _onWheelStart(e) {
    const map = this._map;
    this._isWheeling = true;
    this._wheelMousePosition = map.mouseEventToContainerPoint(e);
    this._centerPoint = map.getSize()._divideBy(2);
    this._startLatLng = map.containerPointToLatLng(this._centerPoint);
    this._wheelStartLatLng = map.containerPointToLatLng(this._wheelMousePosition);
    this._moved = false;
    map._stop();
    this._goalZoom = map.getZoom();
    this._prevCenter = map.getCenter();
    this._prevZoom = map.getZoom();
    this._zoomAnimationId = requestAnimationFrame(() => this._updateWheelZoom());
  },
  _onWheeling(e) {
    const map = this._map;
    const sensitivity = (map.options.smoothSensitivity || 1) * 0.0022;
    this._goalZoom = map._limitZoom(this._goalZoom + L.DomEvent.getWheelDelta(e) * sensitivity);
    this._wheelMousePosition = map.mouseEventToContainerPoint(e);
    window.clearTimeout(this._timeoutId);
    this._timeoutId = window.setTimeout(() => this._onWheelEnd(), 180);
    L.DomEvent.preventDefault(e);
    L.DomEvent.stopPropagation(e);
  },
  _onWheelEnd() {
    this._isWheeling = false;
    cancelAnimationFrame(this._zoomAnimationId);
    this._map._moveEnd(true);
  },
  _updateWheelZoom() {
    const map = this._map;
    // Bail if something else moved the map (don't fight other interactions).
    if (!map.getCenter().equals(this._prevCenter) || map.getZoom() !== this._prevZoom) return;
    this._zoom = Math.round((map.getZoom() + (this._goalZoom - map.getZoom()) * 0.3) * 100) / 100;
    const delta = this._wheelMousePosition.subtract(this._centerPoint);
    this._center =
      map.options.smoothWheelZoom === "center" || (delta.x === 0 && delta.y === 0)
        ? this._startLatLng
        : map.unproject(map.project(this._wheelStartLatLng, this._zoom).subtract(delta), this._zoom);
    if (!this._moved) {
      map._moveStart(true, false);
      this._moved = true;
    }
    map._move(this._center, this._zoom);
    this._prevCenter = map.getCenter();
    this._prevZoom = map.getZoom();
    this._zoomAnimationId = requestAnimationFrame(() => this._updateWheelZoom());
  },
});
L.Map.mergeOptions({ smoothWheelZoom: true, smoothSensitivity: 1.1 });
L.Map.addInitHook("addHandler", "smoothWheelZoom", SmoothWheelZoom);

async function main() {
  const map = L.map("map", {
    center: CENTER,
    zoom: INITIAL_ZOOM,
    minZoom: 7,
    maxZoom: 16,
    zoomControl: false,
    attributionControl: false,
    // Google-Maps-style smooth zoom: continuous (fractional) levels with progressive tile
    // fade-in. Wheel/trackpad zoom is handled by the custom SmoothWheelZoom handler (glides
    // toward the cursor), so the default stepped scrollWheelZoom is disabled. Touch pinch keeps
    // Leaflet's native continuous behaviour.
    fadeAnimation: true,
    zoomSnap: 0,
    zoomDelta: 0.8,
    scrollWheelZoom: false,
    smoothWheelZoom: true,
    smoothSensitivity: 1.1,
  });

  L.tileLayer(BASEMAP_TILE_URL, {
    maxZoom: 19,
    keepBuffer: 4,
    updateWhenZooming: true,
    updateInterval: 80,
  }).addTo(map);

  const zoomControl = L.control.zoom({ position: "topleft" }).addTo(map);

  const payload = await fetchInitialForecastPayload();
  if (!payload) throw new Error("Unable to load any Wind2D forecast model");
  const overlay = new AromeWindOverlay(payload);
  const primaryLayer = rawLayerKeyForPayload(payload) || overlay.primaryRawLayer;
  const primaryModel = rawModelForKey(overlay, primaryLayer);
  if (primaryModel) overlay.rawLayerPayloadSignatures[primaryLayer] = modelPayloadSignature(primaryModel);
  overlay.stepIndex = chooseInitialForecastIndex(payload);
  overlay.activeLeadHour = Number(payload.forecast_steps[overlay.stepIndex]?.lead_hour ?? overlay.activeLeadHour);
  applyPreferredForecastLayer(overlay);
  zoomControl.getContainer()?.addEventListener("pointerdown", () => overlay.prepareZoomInput(), {
    capture: true,
    passive: true,
  });
  map.getContainer()?.addEventListener("wheel", () => overlay.prepareZoomInput(110), {
    capture: true,
    passive: true,
  });
  window.CORSEWIND_AROME_OVERLAY = overlay;
  window.CORSEWIND_AROME_PAYLOAD = payload;
  overlay.addTo(map);
  map.fitBounds(overlay.bounds(), { paddingTopLeft: [16, 100], paddingBottomRight: [16, 84] });
  bindScaleControl(overlay);
  bindModeControl(overlay);
  bindLayerControl(overlay, payload);
  bindParticleControl(overlay);
  bindParticleSliders(overlay);
  bindMapFocusControl();
  bindLegendCompactControl();
  buildForecastButtons(payload, overlay);
  bindMapReadout(map, overlay);
  updateReadout(payload, overlay);
  scheduleOptionalRawModelPreload(overlay);
  window.setTimeout(() => {
    loadModelRasterManifests(overlay).catch((error) => {
      console.debug("Model raster manifest load failed", error);
    });
    refreshWindNinja50mManifest(payload, overlay).catch((error) => {
      console.debug("WindNinja initial manifest refresh failed", error);
    });
  }, 0);
  startProgressiveWindNinjaPolling(payload, overlay);
  startRawModelUpdatePolling(overlay);
}

async function fetchGzipJson(url, bustCache = false) {
  if (!url.includes(".json")) return null;
  const gzipUrl = gzipJsonUrl(url);
  if (gzipUrl === url) return null;
  const response = await fetch(bustCache ? cacheBustedUrl(gzipUrl) : gzipUrl, { cache: bustCache ? "no-store" : "default" });
  if (!response.ok) return null;

  const contentEncoding = response.headers.get("Content-Encoding") || "";
  if (contentEncoding.toLowerCase().includes("gzip")) {
    return await response.json();
  }
  if (!response.body || !("DecompressionStream" in window)) return null;

  const decompressed = response.body.pipeThrough(new DecompressionStream("gzip"));
  return await new Response(decompressed).json();
}

async function fetchJsonWithGzipFallback(url, bustCache = false) {
  try {
    const compressed = await fetchGzipJson(url, bustCache);
    if (compressed) return compressed;
  } catch (error) {
    console.debug("Compressed JSON fetch failed, falling back to plain JSON", error);
  }
  const response = await fetch(bustCache ? cacheBustedUrl(url) : url, { cache: bustCache ? "no-store" : "default" });
  if (!response.ok) return null;
  return await response.json();
}

async function fetchInitialForecastPayload() {
  for (const candidate of INITIAL_MODEL_URLS) {
    try {
      const payload = await fetchJsonWithGzipFallback(candidate.url);
      if (payload?.forecast_steps?.length) return payload;
    } catch (error) {
      console.debug(`Initial ${rawLayerLabel(candidate.layer)} load failed`, error);
    }
  }
  return null;
}

async function fetchOptionalJson(url, bustCache = false, gzipFirst = false) {
  try {
    if (gzipFirst) return await fetchJsonWithGzipFallback(url, bustCache);
    const response = await fetch(bustCache ? cacheBustedUrl(url) : url, { cache: bustCache ? "no-store" : "default" });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

main().catch((error) => {
  console.error(error);
  document.querySelector("#spot-detail").textContent = "Erreur de chargement du champ météo";
});
