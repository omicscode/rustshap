use plotters::prelude::*;

/*
Gaurav Sablok
codeprog@icloud.com
 added a plotters function
*/

fn plot_shap(shap: &Array4<f32>, pos: usize) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("shap_heatmap.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let values: Vec<Vec<f32>> = shap
        .slice(s![0, pos, .., ..])
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect();

    let mut chart = ChartBuilder::on(&root)
        .caption("SHAP Values", ("sans-serif", 20))
        .build_cartesian_2d(0..4, 0..10)?;

    chart.draw_series(values.iter().enumerate().flat_map(|(y, row)| {
        row.iter().enumerate().map(move |(x, &val)| {
            Rectangle::new(
                [(x, y), (x + 1, y + 1)],
                if val > 0.0 {
                    RED.mix(val.abs())
                } else {
                    BLUE.mix(val.abs())
                }
                .filled(),
            )
        })
    }))?;

    Ok(())
}
