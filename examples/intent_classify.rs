// Intent classification benchmark: "is this message a question to answer, or a
// note to remember?"
//
// Measures the on-device cost and accuracy of a SHORT structured-output call —
// the candidate mechanism for answer-vs-ingest routing in the Android app.
// Unlike `simple_chat`, the prompt carries no retrieved context: just a few
// lines of instruction, a tiny JSON schema, and the user's message.
//
// Usage:
//   cargo build --example intent_classify --features clap,serde_json \
//       --target aarch64-linux-android --release --no-default-features
//   ./intent_classify --model-path model.litertlm --backend cpu \
//       --cases-file cases.tsv --add-constraint
//
// cases.tsv is one case per line: <expected>\t<message>, where <expected> is
// `question` or `note`. Lines starting with '#' are skipped.

use std::fs;
use std::process;
use std::time::Instant;

use clap::Parser;
use serde_json::Value;

use cognee_litert_lm::{
    Backend, ConstraintType, Conversation, ConversationConfig, Engine, EngineSettings, OptionalArgs,
    SessionConfig,
};

#[derive(Parser)]
#[command(name = "intent_classify")]
struct Args {
    /// Path to the .litertlm model file.
    #[arg(long)]
    model_path: String,

    /// Backend to use (cpu, gpu).
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Enable JSON-schema constrained decoding (what the cognee adapter does).
    #[arg(long, default_value_t = false)]
    add_constraint: bool,

    /// TSV file of `<expected>\t<message>` cases.
    #[arg(long, default_value = "")]
    cases_file: String,

    /// A single message to classify (instead of --cases-file).
    #[arg(long, default_value = "")]
    message: String,

    /// Repeat the whole case list this many times (to see run-to-run spread).
    #[arg(long, default_value_t = 1)]
    repeats: usize,

    /// Cap on generated tokens for one classification.
    #[arg(long, default_value_t = 64)]
    max_output_tokens: i32,

    /// Use a schema without an `enum` (probe llguidance grammar support).
    #[arg(long, default_value_t = false)]
    plain_schema: bool,

    /// Prompt variant: `adapter` (schema pasted in, what the SDK does today)
    /// or `example` (one literal example of the answer instead).
    #[arg(long, default_value = "adapter")]
    variant: String,
}

/// The schemars-shaped schema a `#[derive(JsonSchema)]` classifier struct would
/// produce. `--plain-schema` drops the enum, to test whether llguidance's
/// grammar can take it.
const SCHEMA_ENUM: &str = r#"{"title":"IntentClassification","type":"object","required":["intent"],"properties":{"intent":{"type":"string","enum":["question","note"]}}}"#;
const SCHEMA_PLAIN: &str = r#"{"title":"IntentClassification","type":"object","required":["intent"],"properties":{"intent":{"type":"string"}}}"#;

const SYSTEM_BIASED: &str = "You route one message a user typed into a personal memory assistant.\n\nSet intent to \"question\" when the user wants something back: a question, a request to look something up, or anything they expect an answer to.\nSet intent to \"note\" when the user is only telling the assistant something to remember: it states a fact and asks for nothing.\n\nIf the message could be either, answer \"question\".";

const SYSTEM_NEUTRAL: &str = "You route one message a user typed into a personal memory assistant.\n\nSet intent to \"note\" when the user is recording something for later: the message states a fact, a decision or a detail, and asks for nothing.\nSet intent to \"question\" when the user wants an answer now.";

const SYSTEM_STRICT: &str = "You route one message a user typed into a personal memory assistant.\n\nSet intent to \"note\" ONLY when the message contains no question and no request of any kind -- it just states a fact, a decision or a detail for the assistant to keep.\n\nIf the message contains a question, a request, an instruction to do something, or anything the user expects a reply to, set intent to \"question\".";

const SYSTEM_FEWSHOT: &str = "Decide what one message typed into a personal memory assistant is for.\n\n\"note\" - the user is recording something for later. It states a fact, a decision or a detail and asks for nothing.\n\"question\" - the user wants an answer now.\n\nExamples:\nThe shop opens at nine. -> {\"intent\":\"note\"}\nWhat time does the shop open? -> {\"intent\":\"question\"}\nSam is joining the call tomorrow. -> {\"intent\":\"note\"}\nWho is on the call tomorrow? -> {\"intent\":\"question\"}\nWe moved the release to Friday. -> {\"intent\":\"note\"}\nWhen is the release? -> {\"intent\":\"question\"}";

fn system_for(variant: &str) -> &'static str {
    match variant {
        "neutral" | "swapped" | "yesno" | "yesno_swapped" | "clean" => SYSTEM_NEUTRAL,
        "strict" => SYSTEM_STRICT,
        "fewshot" | "clean_fewshot" => SYSTEM_FEWSHOT,
        _ => SYSTEM_BIASED,
    }
}

/// Exactly what `LiteRtAdapter::structured_output_impl` appends to the last
/// user message, then what `build_prompt` makes of the message list.
fn build_prompt(message: &str, schema: &str, variant: &str) -> String {
    let instructions = match variant {
        "adapter" => format!(
            "\n\nReturn ONLY a valid JSON object that matches this schema (compact JSON):\n{schema}\nNo markdown, no explanation, and no surrounding text."
        ),
        // Same as `neutral`, but the two literals appear in the other order.
        "swapped" | "strict" => "\n\nAnswer with exactly one JSON object, either {\"intent\":\"question\"} or {\"intent\":\"note\"}. Output that object and nothing else: no schema, no markdown, no explanation."
            .to_string(),
        // The decision is a boolean, so neither label can be copied.
        "yesno" => "\n\nDoes this message ask for an answer now? Reply with exactly {\"answer_now\":true} or {\"answer_now\":false} and nothing else."
            .to_string(),
        "yesno_swapped" => "\n\nIs this message only a note to store? Reply with exactly {\"answer_now\":false} or {\"answer_now\":true} and nothing else."
            .to_string(),
        _ => "\n\nAnswer with exactly one JSON object, either {\"intent\":\"note\"} or {\"intent\":\"question\"}. Output that object and nothing else: no schema, no markdown, no explanation."
            .to_string(),
    };
    let system = system_for(variant);
    // `clean` drops the adapter's fabricated role labels and its trailing
    // "Assistant:" cue, leaving the chat template to do its own job.
    if variant.starts_with("clean") {
        return format!("{system}\n\nMessage: {message}{instructions}");
    }
    format!("System:\n{system}\n\nUser:\n{message}{instructions}\n\nAssistant:\n")
}

/// Salvage the first {...} block, the way the adapter's repair heuristics do.
fn extract_first_json_object(text: &str) -> &str {
    let mut depth = 0i32;
    let mut start = None;
    for (i, c) in text.char_indices() {
        match c {
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

fn parse_intent(raw: &str) -> Option<String> {
    let candidate = match serde_json::from_str::<Value>(raw) {
        Ok(v) => v,
        Err(_) => serde_json::from_str::<Value>(extract_first_json_object(raw)).ok()?,
    };
    if let Some(v) = candidate.get("answer_now").and_then(|v| v.as_bool()) {
        return Some(if v { "question" } else { "note" }.to_string());
    }
    candidate
        .get("intent")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_lowercase())
}

struct Outcome {
    expected: String,
    predicted: String,
    raw: String,
    millis: u128,
    prefill_tokens: i32,
    decode_tokens: i32,
}

fn classify(engine: &Engine, args: &Args, message: &str) -> Result<(String, String, u128, i32, i32), String> {
    let mut session_config = SessionConfig::new().map_err(|e| format!("SessionConfig::new: {e}"))?;
    session_config.set_max_output_tokens(args.max_output_tokens);

    let conversation_config = ConversationConfig::new(
        engine,
        Some(&session_config),
        None,
        None,
        None,
        args.add_constraint,
    )
    .map_err(|e| format!("ConversationConfig::new: {e}"))?;
    let conversation = Conversation::new(engine, Some(conversation_config))
        .map_err(|e| format!("Conversation::new: {e}"))?;

    let schema = if args.plain_schema { SCHEMA_PLAIN } else { SCHEMA_ENUM };
    let message_json = serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": build_prompt(message, schema, &args.variant)}]
    })
    .to_string();

    let optional_args = if args.add_constraint {
        let mut oa = OptionalArgs::new().map_err(|e| format!("OptionalArgs::new: {e}"))?;
        oa.set_constraint(ConstraintType::JsonSchema, schema)
            .map_err(|e| format!("set_constraint: {e}"))?;
        Some(oa)
    } else {
        None
    };

    let started = Instant::now();
    let response = conversation
        .send_message_with_args(&message_json, optional_args.as_ref())
        .map_err(|e| format!("send_message_with_args: {e}"))?;
    let millis = started.elapsed().as_millis();

    let response_str = response.as_str().ok_or("response.as_str() was None")?;
    let mut text = String::new();
    if let Ok(v) = serde_json::from_str::<Value>(response_str) {
        if let Some(content) = v.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                }
            }
        }
    }

    let (mut prefill, mut decode) = (-1, -1);
    if let Ok(b) = conversation.get_benchmark_info() {
        if b.num_prefill_turns() > 0 {
            prefill = b.prefill_token_count_at(0);
        }
        if b.num_decode_turns() > 0 {
            decode = b.decode_token_count_at(0);
        }
    }

    let predicted = parse_intent(&text).unwrap_or_else(|| "UNPARSEABLE".to_string());
    Ok((predicted, text, millis, prefill, decode))
}

fn run() -> i32 {
    let args = Args::parse();
    if args.model_path.is_empty() {
        eprintln!("--model-path is required.");
        return 1;
    }

    let mut cases: Vec<(String, String)> = Vec::new();
    if !args.cases_file.is_empty() {
        let body = match fs::read_to_string(&args.cases_file) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Could not read {}: {e}", args.cases_file);
                return 1;
            }
        };
        for line in body.lines() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match line.split_once('\t') {
                Some((expected, message)) => {
                    cases.push((expected.trim().to_string(), message.trim().to_string()))
                }
                None => eprintln!("skipping malformed line: {line}"),
            }
        }
    } else if !args.message.is_empty() {
        cases.push(("?".to_string(), args.message.clone()));
    } else {
        eprintln!("One of --cases-file or --message is required.");
        return 1;
    }

    cognee_litert_lm::set_min_log_level(3);

    let backend = match args.backend.as_str() {
        "cpu" => Backend::Cpu,
        "gpu" => Backend::Gpu,
        other => Backend::Custom(other.to_string()),
    };

    let engine_started = Instant::now();
    let mut settings = match EngineSettings::new(&args.model_path, backend, None, None) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create engine settings: {e}");
            return 1;
        }
    };
    settings.enable_benchmark();
    let engine = match settings.build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create engine: {e}");
            return 1;
        }
    };
    println!(
        "engine_init_ms\t{}",
        engine_started.elapsed().as_millis()
    );
    println!(
        "prompt_chars\t{}",
        build_prompt(
            "",
            if args.plain_schema { SCHEMA_PLAIN } else { SCHEMA_ENUM },
            &args.variant
        )
        .len()
    );
    println!(
        "constrained\t{}\tbackend\t{}\tvariant\t{}",
        args.add_constraint, args.backend, args.variant
    );
    println!();

    let mut outcomes: Vec<Outcome> = Vec::new();
    for round in 0..args.repeats {
        for (expected, message) in &cases {
            match classify(&engine, &args, message) {
                Ok((predicted, raw, millis, prefill, decode)) => {
                    println!(
                        "round={round}\texp={expected}\tgot={predicted}\tms={millis}\tprefill={prefill}\tdecode={decode}\tmsg={message}\traw={}",
                        raw.replace('\n', "\\n")
                    );
                    outcomes.push(Outcome {
                        expected: expected.clone(),
                        predicted,
                        raw,
                        millis,
                        prefill_tokens: prefill,
                        decode_tokens: decode,
                    });
                }
                Err(why) => {
                    println!("round={round}\texp={expected}\tgot=ERROR\tmsg={message}\twhy={why}");
                    outcomes.push(Outcome {
                        expected: expected.clone(),
                        predicted: "ERROR".to_string(),
                        raw: String::new(),
                        millis: 0,
                        prefill_tokens: -1,
                        decode_tokens: -1,
                    });
                }
            }
        }
    }

    // Summary: confusion matrix + latency distribution.
    println!();
    let mut counts = std::collections::BTreeMap::<(String, String), usize>::new();
    for o in &outcomes {
        *counts
            .entry((o.expected.clone(), o.predicted.clone()))
            .or_insert(0) += 1;
    }
    println!("== confusion (expected -> predicted) ==");
    for ((expected, predicted), n) in &counts {
        println!("{expected}\t->\t{predicted}\t{n}");
    }

    let mut times: Vec<u128> = outcomes.iter().map(|o| o.millis).filter(|m| *m > 0).collect();
    times.sort_unstable();
    if !times.is_empty() {
        let sum: u128 = times.iter().sum();
        println!();
        println!("== latency (ms) ==");
        println!("n\t{}", times.len());
        println!("min\t{}", times[0]);
        println!("median\t{}", times[times.len() / 2]);
        println!("mean\t{}", sum / times.len() as u128);
        println!("max\t{}", times[times.len() - 1]);
    }
    let prefills: Vec<i32> = outcomes
        .iter()
        .map(|o| o.prefill_tokens)
        .filter(|t| *t > 0)
        .collect();
    let decodes: Vec<i32> = outcomes
        .iter()
        .map(|o| o.decode_tokens)
        .filter(|t| *t > 0)
        .collect();
    if !prefills.is_empty() {
        println!(
            "prefill_tokens\tmin={}\tmax={}",
            prefills.iter().min().unwrap(),
            prefills.iter().max().unwrap()
        );
    }
    if !decodes.is_empty() {
        println!(
            "decode_tokens\tmin={}\tmax={}",
            decodes.iter().min().unwrap(),
            decodes.iter().max().unwrap()
        );
    }
    let unparseable = outcomes
        .iter()
        .filter(|o| o.predicted == "UNPARSEABLE" || o.predicted == "ERROR")
        .count();
    println!("unparseable_or_error\t{unparseable}");
    0
}

fn main() {
    process::exit(run());
}
