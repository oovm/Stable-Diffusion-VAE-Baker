//! Local HTTP task service. Model execution is injected through `DiffusionPipeline`.
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use diffusion_types::{CancellationToken, DiffusionPipeline, GenerationRequest, ProgressSink};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;
#[derive(Clone)]
pub struct AppState {
    pub tasks: Arc<RwLock<HashMap<Uuid, TaskStatus>>>,
    pub pipeline: Option<Arc<dyn DiffusionPipeline>>,
    pub mode: String,
}
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", content = "detail")]
pub enum TaskStatus {
    Queued,
    Running,
    Completed { seed: u64 },
    Failed(String),
    Cancelled,
}
#[derive(Debug, Deserialize)]
pub struct ImageRequest {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default = "default_cfg")]
    pub cfg_scale: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub extra_body: Option<serde_json::Value>,
}
fn default_size() -> String {
    "512x512".into()
}
fn default_steps() -> u32 {
    20
}
fn default_cfg() -> f32 {
    7.5
}
pub fn router(state: AppState, pages_dir: std::path::PathBuf) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/api/runtime", get(runtime))
        .route("/v1/images/generations", post(generate))
        .route("/v1/tasks/:id", get(task))
        .fallback_service(ServeDir::new(pages_dir).append_index_html_on_directories(true))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
async fn runtime(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({"mode": state.mode, "backend": "sd", "local": true}))
}
struct NoProgress;
impl ProgressSink for NoProgress {
    fn step(&self, _: u32, _: u32) {}
}
async fn generate(State(state): State<AppState>, Json(input): Json<ImageRequest>) -> impl IntoResponse {
    let id = Uuid::new_v4();
    if input.extra_body.is_some() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({"error":"extra_body extensions are not enabled in this build"})),
        );
    }
    let (width, height) = match input.size.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?))) {
        Some(v) => v,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"size must be WIDTHxHEIGHT"}))),
    };
    state.tasks.write().await.insert(id, TaskStatus::Queued);
    let request = GenerationRequest {
        prompt: input.prompt,
        negative_prompt: input.negative_prompt,
        width,
        height,
        steps: input.steps,
        guidance_scale: input.cfg_scale,
        seed: input.seed,
        loras: vec![],
        embeddings: vec![],
        conditioning: None,
        controlnets: vec![],
        annotator: None,
        ..Default::default()
    };
    if let Err(error) = request.validate() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": error.to_string()})));
    }
    let pipeline = match state.pipeline.clone() {
        Some(pipeline) => pipeline,
        None => {
            state.tasks.write().await.insert(id, TaskStatus::Failed("no pipeline configured".into()));
            return (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({"id": id, "error":"no pipeline configured"})));
        }
    };
    let tasks = state.tasks.clone();
    tokio::task::spawn_blocking(move || {
        tasks.blocking_write().insert(id, TaskStatus::Running);
        let status = match pipeline.generate(&request, &NoProgress, &CancellationToken::default()) {
            Ok(result) => TaskStatus::Completed { seed: result.seed },
            Err(error) => TaskStatus::Failed(error.to_string()),
        };
        tasks.blocking_write().insert(id, status);
    });
    (StatusCode::ACCEPTED, Json(serde_json::json!({"id": id, "status":"queued"})))
}
async fn task(State(state): State<AppState>, Path(id): Path<Uuid>) -> impl IntoResponse {
    match state.tasks.read().await.get(&id) {
        Some(status) => (StatusCode::OK, Json(serde_json::to_value(status).unwrap())),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"unknown task"}))),
    }
}
pub async fn serve(state: AppState, address: std::net::SocketAddr, pages_dir: std::path::PathBuf) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, router(state, pages_dir)).await.map_err(std::io::Error::other)
}
pub fn empty_state() -> AppState {
    AppState { tasks: Arc::new(RwLock::new(HashMap::new())), pipeline: None, mode: "local".into() }
}
