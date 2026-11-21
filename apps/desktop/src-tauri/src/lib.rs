use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let navigation = tauri::plugin::Builder::<_, ()>::new("studio-navigation")
        .on_navigation(|view, url| {
            if tauri::is_dev() {
                if let Some(studio) = &view.app_handle().config().build.dev_url {
                    return url.origin() == studio.origin();
                }
            }
            true
        })
        .build();
    tauri::Builder::default()
        .plugin(navigation)
        .run(tauri::generate_context!())
        .expect("error while running Video to 3D");
}
