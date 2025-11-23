use plotters::prelude::*;

/*
Gaurav Sablok
codeprog@icloud.com
*/

pub fn shapdeepmap(
    shap: &Array4<f32>,
    pos: usize,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let data: Vec<Vec<f32>> = shap
        .slice(s![0, pos, .., ..])
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect();

    let max_abs = data.iter().flatten().map(|v| v.abs()).fold(0.0, f32::max);
    let norm = |v| (v / max_abs.max(1e-8)) * 0.8 + 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("DeepSHAP - Position {}", pos), ("sans-serif", 20))
        .build_cartesian_2d(0..4, 0..10)?;

    chart.draw_series(data.iter().enumerate().flat_map(|(y, row)| {
        row.iter().enumerate().map(move |(x, &val)| {
            let color = if val > 0.0 {
                RED.mix(norm(val))
            } else {
                BLUE.mix(norm(-val))
            };
            Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
        })
    }))?;

    Ok(())
}
