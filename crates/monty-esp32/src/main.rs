//! Monty Python interpreter for the M5Stack Cardputer (ESP32-S3).
//!
//! Loads pre-compiled Monty bytecode and executes it on the device,
//! rendering output to the 240x135 LCD display.
//!
//! # Usage
//!
//! 1. Compile Python to bytecode on a host machine:
//!    ```bash
//!    cargo run -p monty-cli -- compile script.py -o script.monty
//!    ```
//!
//! 2. Either embed the bytecode in the binary with `include_bytes!()`,
//!    or load it from the SD card at runtime.
//!
//! 3. Build and flash:
//!    ```bash
//!    cargo +esp build --release --target xtensa-esp32s3-espidf
//!    espflash flash --chip esp32s3 target/xtensa-esp32s3-espidf/release/monty-esp32
//!    ```

use std::borrow::Cow;
use std::time::Duration;

use monty::{
    CollectStringPrint, LimitedTracker, MontyException, MontyRun, PrintWriter, ResourceLimits,
};

/// Embedded bytecode for a demo program.
///
/// Replace this with your own compiled bytecode, or load from SD card.
/// Compile with: `cargo run -p monty-cli -- compile hello.py -o hello.monty`
const DEMO_BYTECODE: &[u8] = include_bytes!("../demo.monty");

fn main() {
    // Initialize ESP-IDF logging
    esp_idf_svc::log::EspLogger::initialize_default();
    log::info!("Monty ESP32 starting...");

    // Load the pre-compiled bytecode
    let runner = match MontyRun::load(DEMO_BYTECODE) {
        Ok(r) => r,
        Err(err) => {
            log::error!("Failed to load bytecode: {err}");
            panic!("bytecode load failed");
        }
    };

    // Configure resource limits for ESP32 (no PSRAM — 512KB SRAM total,
    // ~200KB usable heap after ESP-IDF, stack, and static allocations)
    let limits = ResourceLimits::new()
        .max_memory(128 * 1024) // 128 KB heap limit
        .max_duration(Duration::from_secs(30)) // 30 second timeout
        .max_recursion_depth(Some(100)) // conservative for 32KB stack
        .gc_interval(10_000);
    let tracker = LimitedTracker::new(limits);

    // Use a string collector for now — will be replaced with LCD output
    let mut print = CollectStringPrint::new();

    // Run the bytecode
    log::info!("Executing bytecode...");
    match runner.run(vec![], tracker, &mut print) {
        Ok(result) => {
            let output = print.output();
            log::info!("Output:\n{output}");
            log::info!("Result: {result}");
        }
        Err(err) => {
            log::error!("Runtime error:\n{err}");
        }
    }

    log::info!("Monty ESP32 done.");

    // TODO: Replace the above with a main loop that:
    // 1. Initializes the ST7789 display via SPI
    // 2. Scans the keyboard matrix for input
    // 3. Lists .monty files from the SD card
    // 4. Lets the user select and run programs
    // 5. Renders output to the LCD
}

/// PrintWriter implementation that renders text to the Cardputer's LCD display.
///
/// Maintains a scrolling text buffer and renders it to the ST7789 display
/// using embedded-graphics. The display is 240x135 pixels, fitting approximately
/// 30 columns x 11 lines with an 8x12 pixel monospace font.
struct LcdPrintWriter {
    /// Lines of text output (circular buffer for scroll-back).
    lines: Vec<String>,
    /// Current line being built (not yet terminated by newline).
    current_line: String,
    // TODO: Add display handle when display driver is integrated
}

#[expect(dead_code)]
impl LcdPrintWriter {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            current_line: String::new(),
        }
    }
}

impl PrintWriter for LcdPrintWriter {
    fn stdout_write(&mut self, output: Cow<'_, str>) -> Result<(), MontyException> {
        self.current_line.push_str(&output);
        // TODO: Render to display
        Ok(())
    }

    fn stdout_push(&mut self, end: char) -> Result<(), MontyException> {
        if end == '\n' {
            self.lines.push(std::mem::take(&mut self.current_line));
            // TODO: Scroll display and render new line
        } else {
            self.current_line.push(end);
        }
        Ok(())
    }
}
