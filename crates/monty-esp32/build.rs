/// ESP-IDF build system integration.
///
/// This delegates to embuild which handles the ESP-IDF CMake build,
/// linking, and environment variable setup.
fn main() {
    embuild::espidf::sysenv::output();
}
