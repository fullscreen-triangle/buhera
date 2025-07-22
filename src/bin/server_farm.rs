//! # Buhera VPOS Gas Oscillation Server Farm
//!
//! This binary demonstrates the complete server farm implementation with
//! consciousness substrate, gas oscillation processors, and zero-cost cooling.

use std::time::Duration;
use clap::{Arg, Command};
use tracing::{info, error};
use tokio::signal;

use buhera::server_farm::{
    ConsciousnessSubstrate, ConsciousnessConfig,
    CoolingSystemManager, CoolingConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("buhera=debug,server_farm=debug")
        .init();

    // Parse command line arguments
    let matches = Command::new("buhera-server-farm")
        .version("0.1.0")
        .author("Buhera Research Team")
        .about("Buhera VPOS Gas Oscillation Server Farm with Consciousness Substrate")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("config/server_farm/default.toml")
        )
        .arg(
            Arg::new("mode")
                .short('m')
                .long("mode")
                .value_name("MODE")
                .help("Operating mode")
                .default_value("production")
                .value_parser(["development", "production", "test"])
        )
        .arg(
            Arg::new("consciousness")
                .long("enable-consciousness")
                .help("Enable consciousness substrate")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("cooling")
                .long("enable-zero-cost-cooling")
                .help("Enable zero-cost cooling system")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();

    let config_file = matches.get_one::<String>("config").unwrap();
    let mode = matches.get_one::<String>("mode").unwrap();
    let enable_consciousness = matches.get_flag("consciousness");
    let enable_cooling = matches.get_flag("cooling");

    info!("Starting Buhera VPOS Gas Oscillation Server Farm");
    info!("Mode: {}", mode);
    info!("Config: {}", config_file);
    info!("Consciousness enabled: {}", enable_consciousness);
    info!("Zero-cost cooling enabled: {}", enable_cooling);

    // Initialize server farm components
    let mut server_farm = ServerFarm::new(
        mode.to_string(),
        enable_consciousness,
        enable_cooling,
    ).await?;

    // Start server farm
    server_farm.start().await?;

    // Wait for shutdown signal
    info!("Server farm running. Press Ctrl+C to shutdown.");
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Received shutdown signal, stopping server farm...");
        }
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }

    // Shutdown server farm
    server_farm.shutdown().await?;

    info!("Buhera VPOS Gas Oscillation Server Farm shutdown complete");
    Ok(())
}

/// Server farm orchestrator
pub struct ServerFarm {
    mode: String,
    consciousness_substrate: Option<ConsciousnessSubstrate>,
    cooling_system: Option<CoolingSystemManager>,
}

impl ServerFarm {
    /// Create new server farm
    pub async fn new(
        mode: String,
        enable_consciousness: bool,
        enable_cooling: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut server_farm = Self {
            mode,
            consciousness_substrate: None,
            cooling_system: None,
        };

        // Initialize consciousness substrate if enabled
        if enable_consciousness {
            info!("Initializing consciousness substrate...");
            let consciousness_config = ConsciousnessConfig::default();
            let consciousness = ConsciousnessSubstrate::with_config(consciousness_config)?;
            consciousness.initialize().await?;
            server_farm.consciousness_substrate = Some(consciousness);
            info!("Consciousness substrate initialized successfully");
        }

        // Initialize zero-cost cooling system if enabled
        if enable_cooling {
            info!("Initializing zero-cost cooling system...");
            let cooling_config = CoolingConfig::default();
            let cooling_system = CoolingSystemManager::new(cooling_config)?;
            cooling_system.initialize().await?;
            server_farm.cooling_system = Some(cooling_system);
            info!("Zero-cost cooling system initialized successfully");
        }

        Ok(server_farm)
    }

    /// Start server farm
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting server farm components...");

        // Start consciousness substrate
        if let Some(consciousness) = &self.consciousness_substrate {
            consciousness.start().await?;
            info!("Consciousness substrate started");
        }

        // Start cooling system
        if let Some(cooling) = &self.cooling_system {
            let _cooling_task = cooling.start_cooling().await?;
            info!("Zero-cost cooling system started");
        }

        // Start monitoring and status reporting
        self.start_monitoring().await?;

        info!("All server farm components started successfully");
        Ok(())
    }

    /// Start monitoring
    async fn start_monitoring(&self) -> Result<(), Box<dyn std::error::Error>> {
        let consciousness = self.consciousness_substrate.as_ref();
        let cooling = self.cooling_system.as_ref();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Monitor consciousness substrate
                if let Some(consciousness) = consciousness {
                    let state = consciousness.get_state();
                    info!(
                        "Consciousness: coherence={:.3}, awareness={:.3}, memory={:.3}",
                        state.coherence_level,
                        state.awareness_level,
                        state.memory_utilization
                    );
                }
                
                // Monitor cooling system
                if let Some(cooling) = cooling {
                    let state = cooling.get_state();
                    info!(
                        "Cooling: temp={:.1}K, efficiency={:.3}, energy={:.1}W",
                        state.current_temperature,
                        state.efficiency,
                        state.energy_consumption
                    );
                }
            }
        });

        Ok(())
    }

    /// Shutdown server farm
    pub async fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Shutting down server farm components...");

        // Stop cooling system
        if let Some(cooling) = &self.cooling_system {
            cooling.stop_cooling().await?;
            info!("Zero-cost cooling system stopped");
        }

        // Stop consciousness substrate
        if let Some(consciousness) = &self.consciousness_substrate {
            consciousness.stop().await?;
            info!("Consciousness substrate stopped");
        }

        info!("Server farm shutdown complete");
        Ok(())
    }
} 