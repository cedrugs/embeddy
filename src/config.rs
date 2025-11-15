use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
	pub data_dir: PathBuf,
	pub models_dir: PathBuf,
	pub registry_path: PathBuf,
}

impl Config {
	pub fn new() -> crate::error::Result<Self> {
		let project_dirs = ProjectDirs::from("", "", "embeddy")
			.ok_or_else(|| crate::error::Error::ConfigError("Could not determine config directory".to_string()))?;

		let data_dir = project_dirs.data_dir().to_path_buf();
		let models_dir = data_dir.join("models");
		let registry_path = data_dir.join("models.toml");

		std::fs::create_dir_all(&data_dir)?;
		std::fs::create_dir_all(&models_dir)?;

		Ok(Self {
			data_dir,
			models_dir,
			registry_path,
		})
	}

	pub fn from_env() -> crate::error::Result<Self> {
		if let Ok(data_dir) = std::env::var("EMBEDDY_DATA_DIR") {
			let data_dir = PathBuf::from(data_dir);
			let models_dir = data_dir.join("models");
			let registry_path = data_dir.join("models.toml");

			std::fs::create_dir_all(&data_dir)?;
			std::fs::create_dir_all(&models_dir)?;

			Ok(Self {
				data_dir,
				models_dir,
				registry_path,
			})
		} else {
			Self::new()
		}
	}
}

impl Default for Config {
	fn default() -> Self {
		Self::new().expect("Failed to create default config")
	}
}
