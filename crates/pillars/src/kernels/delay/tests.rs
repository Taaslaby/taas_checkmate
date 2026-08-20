use super::*;

#[test]
fn delay_buffer_circular_basic() {
    let buffer: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; 8];
    delay_buffer_circular(&mut output, &buffer, 3);
    assert_eq!(output, vec![3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0]);
}

#[test]
fn delay_buffer_multi_tap_basic() {
    let buffer: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let delays = vec![0i32, 3, 7, 9];
    let mut output = vec![0.0f32; 4];
    delay_buffer_multi_tap(&mut output, &buffer, &delays);
    assert_eq!(output, vec![0.0, 3.0, 7.0, 9.0]);
}

#[test]
fn delay_buffer_tap_clamped() {
    let buffer: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let delays = vec![100i32];
    let mut output = vec![0.0f32; 1];
    delay_buffer_multi_tap(&mut output, &buffer, &delays);
    assert_eq!(output[0], 3.0, "tap beyond end clamps to last element");
}
