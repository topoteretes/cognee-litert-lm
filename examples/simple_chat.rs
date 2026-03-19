// Knowledge graph extraction demo using the cognee-litert-lm Rust bindings
// with constrained decoding.
//
// Given an input text document, prompts the LLM to extract entities (nodes)
// and relationships (edges) and output them as a JSON object conforming to a
// hardcoded schema.
//
// Usage:
//   cargo run --example simple_chat --features clap,serde_json -- \
//       --model-path /path/to/model.litertlm \
//       --input-text "Alice works at Google. Bob works at Meta." \
//       --backend cpu

use std::fs;
use std::process;

use clap::Parser;
use serde_json::Value;

use cognee_litert_lm::{
    Backend, ConstraintType, Conversation, ConversationConfig, EngineSettings, OptionalArgs,
    SessionConfig,
};

/// Knowledge graph extraction using LiteRT-LM with constrained decoding.
#[derive(Parser)]
#[command(name = "simple_chat")]
struct Args {
    /// Path to the .litertlm model file.
    #[arg(long)]
    model_path: String,

    /// Backend to use (cpu, gpu, npu).
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Enable constrained decoding.
    #[arg(long, default_value_t = false)]
    add_constraint: bool,

    /// Input text to extract knowledge from.
    #[arg(long, default_value = "")]
    input_text: String,

    /// Path to a file containing the input text.
    #[arg(long, default_value = "")]
    input_text_file: String,
}

const KNOWLEDGE_GRAPH_SYSTEM_PROMPT: &str = "\
You are a knowledge graph extraction expert. Your task is to analyze the \
provided text and extract a structured knowledge graph from it.\n\n\
Extract all named entities as nodes with the following properties:\n\
- id: a short snake_case identifier (e.g. 'alice', 'google_inc')\n\
- name: the entity's full display name\n\
- type: the entity type (e.g. 'Person', 'Organization', 'Location', 'Concept')\n\
- description: a brief description of the entity based on the text\n\n\
Extract all relationships as edges with the following properties:\n\
- source_node_id: id of the source node\n\
- target_node_id: id of the target node\n\
- relationship_name: a concise label for the relationship \
(e.g. 'WORKS_AT', 'LOCATED_IN', 'FOUNDED_BY')\n\n\
Output ONLY valid JSON conforming to the provided schema. \
Do not include any explanation or markdown.\n\n\
JSON Schema:\n";

const KNOWLEDGE_GRAPH_SCHEMA: &str = r#"{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "KnowledgeGraph",
  "type": "object",
  "required": ["nodes", "edges"],
  "additionalProperties": false,
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "title": "Node",
        "required": ["id", "name", "type", "description"],
        "additionalProperties": false,
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {"type": "string"},
          "description": {"type": "string"}
        }
      }
    },
    "edges": {
      "type": "array",
      "items": {
        "type": "object",
        "title": "Edge",
        "required": ["source_node_id", "target_node_id", "relationship_name"],
        "additionalProperties": false,
        "properties": {
          "source_node_id": {"type": "string"},
          "target_node_id": {"type": "string"},
          "relationship_name": {"type": "string"}
        }
      }
    }
  }
}"#;

/// Compact a JSON schema string (remove whitespace) for embedding in prompts.
fn compact_json_schema(schema: &str) -> String {
    serde_json::from_str::<Value>(schema)
        .map(|v| v.to_string())
        .unwrap_or_else(|_| schema.to_string())
}

/// Extract the first complete `{...}` JSON object from a string.
fn extract_first_json_object(text: &str) -> &str {
    let mut depth = 0i32;
    let mut start = None;
    for (i, ch) in text.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        return &text[s..=i];
                    }
                }
            }
            _ => {}
        }
    }
    text
}

fn run() -> i32 {
    let args = Args::parse();

    if args.model_path.is_empty() {
        eprintln!("--model-path is required.");
        return 1;
    }
    if args.input_text.is_empty() && args.input_text_file.is_empty() {
        eprintln!("One of --input-text or --input-text-file is required.");
        return 1;
    }
    if !args.input_text.is_empty() && !args.input_text_file.is_empty() {
        eprintln!("Only one of --input-text or --input-text-file may be specified.");
        return 1;
    }

    let source_text = if !args.input_text_file.is_empty() {
        match fs::read_to_string(&args.input_text_file) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Could not open file {}: {e}", args.input_text_file);
                return 1;
            }
        }
    } else {
        args.input_text.clone()
    };

    // Suppress all log output so only the JSON result is printed.
    cognee_litert_lm::set_min_log_level(3); // 3 = FATAL

    // Create engine settings.
    let backend = match args.backend.as_str() {
        "cpu" => Backend::Cpu,
        "gpu" => Backend::Gpu,
        other => Backend::Custom(other.to_string()),
    };

    let mut settings = match EngineSettings::new(&args.model_path, backend, None, None) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create engine settings: {e}");
            return 1;
        }
    };

    // Enable benchmarking.
    settings.enable_benchmark();

    // Create engine.
    let engine = match settings.build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create engine: {e}");
            return 1;
        }
    };

    // Create session config.
    let mut session_config = match SessionConfig::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create session config: {e}");
            return 1;
        }
    };
    session_config.set_max_output_tokens(2048);

    // Create conversation config.
    let conversation_config = match ConversationConfig::new(
        &engine,
        Some(&session_config),
        None, // system_message_json
        None, // tools_json
        None, // messages_json
        args.add_constraint,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create conversation config: {e}");
            return 1;
        }
    };

    // Create conversation.
    let conversation = match Conversation::new(&engine, Some(conversation_config)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create conversation: {e}");
            return 1;
        }
    };

    // Build the prompt: system instructions + compact schema + source text.
    let compact_schema = compact_json_schema(KNOWLEDGE_GRAPH_SCHEMA);
    let prompt = format!(
        "{KNOWLEDGE_GRAPH_SYSTEM_PROMPT}{compact_schema}\n\n\
         Text to extract from:\n{source_text}\n\n\
         Knowledge graph JSON:\n"
    );

    // Create message JSON.
    let message_json = serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    })
    .to_string();

    // Create optional args with constraint if enabled.
    let optional_args = if args.add_constraint {
        let mut oa = match OptionalArgs::new() {
            Ok(oa) => oa,
            Err(e) => {
                eprintln!("Failed to create optional args: {e}");
                return 1;
            }
        };
        if let Err(e) = oa.set_constraint(ConstraintType::JsonSchema, KNOWLEDGE_GRAPH_SCHEMA) {
            eprintln!("Failed to set constraint: {e}");
            return 1;
        }
        Some(oa)
    } else {
        None
    };

    // Send message and get response.
    let response = match conversation.send_message_with_args(&message_json, optional_args.as_ref())
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to send message: {e}");
            return 1;
        }
    };

    let response_str = match response.as_str() {
        Some(s) => s,
        None => {
            eprintln!("Failed to get response string");
            return 1;
        }
    };

    // Parse the response JSON to extract text content.
    let full_response = match serde_json::from_str::<Value>(response_str) {
        Ok(v) => {
            let mut text = String::new();
            if let Some(content) = v.get("content").and_then(|c| c.as_array()) {
                for item in content {
                    if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                        text.push_str(t);
                    }
                }
            }
            text
        }
        Err(e) => {
            eprintln!("Failed to parse response: {e}");
            return 1;
        }
    };

    // Try to parse as JSON; fall back to brace extraction.
    let result_json = match serde_json::from_str::<Value>(&full_response) {
        Ok(v) => serde_json::to_string_pretty(&v).unwrap_or(full_response.clone()),
        Err(_) => {
            let extracted = extract_first_json_object(&full_response);
            match serde_json::from_str::<Value>(extracted) {
                Ok(v) => serde_json::to_string_pretty(&v).unwrap_or(extracted.to_string()),
                Err(_) => full_response.clone(),
            }
        }
    };

    println!("{result_json}");

    // Print benchmark info.
    if let Ok(benchmark) = conversation.get_benchmark_info() {
        let ttft = benchmark.time_to_first_token();
        let init_time = benchmark.total_init_time();
        let num_prefill = benchmark.num_prefill_turns();
        let num_decode = benchmark.num_decode_turns();

        let mut total_prefill_tokens = 0.0_f64;
        let mut total_prefill_time = 0.0_f64;
        for i in 0..num_prefill {
            let tokens = benchmark.prefill_token_count_at(i) as f64;
            let tps = benchmark.prefill_tokens_per_sec_at(i);
            total_prefill_tokens += tokens;
            if tps > 0.0 {
                total_prefill_time += tokens / tps;
            }
        }

        let mut total_decode_tokens = 0.0_f64;
        let mut total_decode_time = 0.0_f64;
        for i in 0..num_decode {
            let tokens = benchmark.decode_token_count_at(i) as f64;
            let tps = benchmark.decode_tokens_per_sec_at(i);
            total_decode_tokens += tokens;
            if tps > 0.0 {
                total_decode_time += tokens / tps;
            }
        }

        let avg_prefill_speed = if total_prefill_time > 0.0 {
            total_prefill_tokens / total_prefill_time
        } else {
            0.0
        };
        let avg_decode_speed = if total_decode_time > 0.0 {
            total_decode_tokens / total_decode_time
        } else {
            0.0
        };
        let total_time = init_time + total_prefill_time + total_decode_time;

        // Output in format expected by benchmark script.
        eprintln!("\nPrefill Speed: {avg_prefill_speed} tokens/sec");
        eprintln!("Decode Speed: {avg_decode_speed} tokens/sec");
        eprintln!("Total Time: {total_time} s");
        eprintln!("Time to first token: {ttft} s");
    }

    0
}

fn main() {
    process::exit(run());
}
