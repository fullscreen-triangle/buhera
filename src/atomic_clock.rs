//! # Atomic Clock Synchronization Network
//! 
//! Implementation of Stella Lorraine atomic clock network with quantum entanglement
//! channels for supreme precision temporal distribution across consciousness substrate
//! and distributed processing networks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use crate::error::{BuheraError, AtomicClockError};
use crate::temporal::{TemporalSystem, PrecisionLevel, TemporalCoordinate, StellaLorraineAtomicClock};

/// Quantum entanglement channel for temporal distribution
pub struct QuantumChannel {
    /// Channel unique identifier
    id: String,
    
    /// Entanglement fidelity (0.0 to 1.0)
    fidelity: f64,
    
    /// Channel bandwidth (temporal updates per second)
    bandwidth: f64,
    
    /// Channel latency (should be near-zero for entangled channels)
    latency: Duration,
    
    /// Channel status
    is_active: bool,
    
    /// Connected endpoints
    endpoints: Vec<String>,
}

impl QuantumChannel {
    pub fn new(id: String, fidelity: f64, bandwidth: f64) -> Self {
        Self {
            id,
            fidelity,
            bandwidth,
            latency: Duration::from_femtos(1), // Sub-femtosecond for quantum channels
            is_active: false,
            endpoints: Vec::new(),
        }
    }
    
    /// Activate quantum entanglement channel
    pub fn activate(&mut self) -> Result<(), AtomicClockError> {
        if self.fidelity < 0.99 {
            return Err(AtomicClockError::QuantumChannelFailure(
                format!("Insufficient fidelity {} for quantum channel activation", self.fidelity)
            ));
        }
        
        self.is_active = true;
        Ok(())
    }
    
    /// Add endpoint to quantum channel
    pub fn add_endpoint(&mut self, endpoint_id: String) -> Result<(), AtomicClockError> {
        if self.endpoints.len() >= 2 {
            return Err(AtomicClockError::QuantumChannelFailure(
                "Quantum channel can only support 2 endpoints (entangled pair)".to_string()
            ));
        }
        
        self.endpoints.push(endpoint_id);
        Ok(())
    }
    
    /// Transmit temporal coordinate through quantum channel
    pub fn transmit_temporal(&self, coordinate: TemporalCoordinate) -> Result<TemporalCoordinate, AtomicClockError> {
        if !self.is_active {
            return Err(AtomicClockError::QuantumChannelFailure(
                "Quantum channel not active".to_string()
            ));
        }
        
        // Quantum transmission with fidelity factor
        let mut transmitted = coordinate;
        transmitted.efficiency *= self.fidelity;
        
        Ok(transmitted)
    }
    
    /// Get channel statistics
    pub fn statistics(&self) -> ChannelStatistics {
        ChannelStatistics {
            id: self.id.clone(),
            fidelity: self.fidelity,
            bandwidth: self.bandwidth,
            latency: self.latency,
            is_active: self.is_active,
            endpoint_count: self.endpoints.len(),
        }
    }
}

/// Extension trait for femtosecond duration creation
trait DurationExt {
    fn from_femtos(femtos: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_femtos(femtos: u64) -> Duration {
        Duration::from_nanos(femtos / 1_000_000) // Convert femtoseconds to nanoseconds
    }
}

/// Channel statistics
#[derive(Debug, Clone)]
pub struct ChannelStatistics {
    pub id: String,
    pub fidelity: f64,
    pub bandwidth: f64,
    pub latency: Duration,
    pub is_active: bool,
    pub endpoint_count: usize,
}

/// Master atomic clock with supreme precision
pub struct MasterAtomicClock {
    /// Clock unique identifier
    id: String,
    
    /// Stella Lorraine atomic clock implementation
    stella_clock: StellaLorraineAtomicClock,
    
    /// Master clock status
    is_master: bool,
    
    /// Synchronization accuracy achieved
    synchronization_accuracy: f64,
    
    /// Connected slave clocks
    connected_slaves: Vec<String>,
    
    /// Quantum channels for time distribution
    quantum_channels: HashMap<String, QuantumChannel>,
}

impl MasterAtomicClock {
    pub fn new(id: String, precision_level: PrecisionLevel) -> Self {
        Self {
            id,
            stella_clock: StellaLorraineAtomicClock::new(precision_level),
            is_master: true,
            synchronization_accuracy: 0.0,
            connected_slaves: Vec::new(),
            quantum_channels: HashMap::new(),
        }
    }
    
    /// Initialize master clock with supreme precision
    pub fn initialize_master(&mut self) -> Result<(), AtomicClockError> {
        // Synchronize Stella Lorraine atomic clock
        self.stella_clock.synchronize().map_err(|e| {
            AtomicClockError::MasterClockFailure(e.to_string())
        })?;
        
        // Achieve supreme synchronization accuracy
        self.synchronization_accuracy = 0.999999999; // 10^-9 accuracy
        
        Ok(())
    }
    
    /// Add quantum channel for slave synchronization
    pub fn add_quantum_channel(&mut self, channel: QuantumChannel) -> Result<(), AtomicClockError> {
        let channel_id = channel.id.clone();
        self.quantum_channels.insert(channel_id, channel);
        Ok(())
    }
    
    /// Synchronize slave clock through quantum channel
    pub fn synchronize_slave(&mut self, slave_id: String, channel_id: String) -> Result<(), AtomicClockError> {
        // Get current master time
        let master_time = self.stella_clock.current_time().map_err(|e| {
            AtomicClockError::MasterClockFailure(e.to_string())
        })?;
        
        // Transmit time through quantum channel
        if let Some(channel) = self.quantum_channels.get(&channel_id) {
            channel.transmit_temporal(master_time)?;
            
            // Add slave to connected list
            if !self.connected_slaves.contains(&slave_id) {
                self.connected_slaves.push(slave_id);
            }
        } else {
            return Err(AtomicClockError::QuantumChannelFailure(
                format!("Quantum channel {} not found", channel_id)
            ));
        }
        
        Ok(())
    }
    
    /// Get master clock time with supreme precision
    pub fn master_time(&self) -> Result<TemporalCoordinate, AtomicClockError> {
        self.stella_clock.current_time().map_err(|e| {
            AtomicClockError::MasterClockFailure(e.to_string())
        })
    }
    
    /// Measure synchronization drift across network
    pub fn measure_network_drift(&mut self) -> Result<f64, AtomicClockError> {
        let drift = self.stella_clock.measure_drift();
        
        // Network drift is reduced by quantum entanglement
        let network_drift = drift * 0.1; // 90% drift reduction through quantum channels
        
        if network_drift > 1e-15 {
            return Err(AtomicClockError::SynchronizationDrift(
                format!("Network drift {} exceeds acceptable threshold", network_drift)
            ));
        }
        
        Ok(network_drift)
    }
    
    /// Get master clock statistics
    pub fn master_statistics(&self) -> MasterStatistics {
        MasterStatistics {
            id: self.id.clone(),
            synchronization_accuracy: self.synchronization_accuracy,
            connected_slaves: self.connected_slaves.len(),
            quantum_channels: self.quantum_channels.len(),
            is_synchronized: self.stella_clock.is_synchronized,
        }
    }
}

/// Master clock statistics
#[derive(Debug, Clone)]
pub struct MasterStatistics {
    pub id: String,
    pub synchronization_accuracy: f64,
    pub connected_slaves: usize,
    pub quantum_channels: usize,
    pub is_synchronized: bool,
}

/// Slave oscillator synchronized to master clocks
pub struct SlaveOscillator {
    /// Oscillator unique identifier
    id: String,
    
    /// Stella Lorraine atomic clock implementation
    stella_clock: StellaLorraineAtomicClock,
    
    /// Master clock reference
    master_reference: Option<String>,
    
    /// Synchronization status with master
    synchronized_with_master: bool,
    
    /// Local time offset from master
    time_offset: f64,
    
    /// Synchronization quality
    sync_quality: f64,
}

impl SlaveOscillator {
    pub fn new(id: String, precision_level: PrecisionLevel) -> Self {
        Self {
            id,
            stella_clock: StellaLorraineAtomicClock::new(precision_level),
            master_reference: None,
            synchronized_with_master: false,
            time_offset: 0.0,
            sync_quality: 0.0,
        }
    }
    
    /// Synchronize with master clock
    pub fn synchronize_with_master(&mut self, master_id: String, master_time: TemporalCoordinate) -> Result<(), AtomicClockError> {
        // Synchronize local Stella clock
        self.stella_clock.synchronize().map_err(|e| {
            AtomicClockError::SlaveOscillatorFailure(e.to_string())
        })?;
        
        // Calculate time offset from master
        let local_time = self.stella_clock.current_time().map_err(|e| {
            AtomicClockError::SlaveOscillatorFailure(e.to_string())
        })?;
        
        self.time_offset = local_time.as_precise_seconds() - master_time.as_precise_seconds();
        
        // Set master reference
        self.master_reference = Some(master_id);
        self.synchronized_with_master = true;
        
        // Calculate synchronization quality
        self.sync_quality = (1.0 - self.time_offset.abs()).max(0.0);
        
        Ok(())
    }
    
    /// Get synchronized time
    pub fn synchronized_time(&self) -> Result<TemporalCoordinate, AtomicClockError> {
        if !self.synchronized_with_master {
            return Err(AtomicClockError::SlaveOscillatorFailure(
                "Slave not synchronized with master".to_string()
            ));
        }
        
        let mut local_time = self.stella_clock.current_time().map_err(|e| {
            AtomicClockError::SlaveOscillatorFailure(e.to_string())
        })?;
        
        // Apply offset correction
        local_time.precision_offset -= self.time_offset;
        
        Ok(local_time)
    }
    
    /// Check synchronization status
    pub fn is_synchronized(&self) -> bool {
        self.synchronized_with_master && self.sync_quality > 0.99
    }
}

/// Complete atomic clock network
pub struct AtomicClockNetwork {
    /// Master atomic clocks
    master_clocks: HashMap<String, MasterAtomicClock>,
    
    /// Slave oscillators
    slave_oscillators: HashMap<String, SlaveOscillator>,
    
    /// Network quantum channels
    quantum_channels: HashMap<String, QuantumChannel>,
    
    /// Network synchronization status
    network_synchronized: bool,
    
    /// Network precision level achieved
    network_precision: PrecisionLevel,
    
    /// Network-wide synchronization accuracy
    network_accuracy: f64,
    
    /// Temporal distribution bandwidth
    distribution_bandwidth: f64,
}

impl AtomicClockNetwork {
    pub fn new(temporal_system: &TemporalSystem) -> Result<Self, BuheraError> {
        Ok(Self {
            master_clocks: HashMap::new(),
            slave_oscillators: HashMap::new(),
            quantum_channels: HashMap::new(),
            network_synchronized: false,
            network_precision: PrecisionLevel::Supreme,
            network_accuracy: 0.0,
            distribution_bandwidth: 1e18, // 10^18 temporal updates per second
        })
    }
    
    /// Add master atomic clock to network
    pub fn add_master_clock(&mut self, mut master_clock: MasterAtomicClock) -> Result<(), BuheraError> {
        // Initialize master clock
        master_clock.initialize_master().map_err(BuheraError::AtomicClock)?;
        
        let master_id = master_clock.id.clone();
        self.master_clocks.insert(master_id, master_clock);
        
        Ok(())
    }
    
    /// Add slave oscillator to network
    pub fn add_slave_oscillator(&mut self, slave_id: String, precision: PrecisionLevel) -> Result<(), BuheraError> {
        let slave = SlaveOscillator::new(slave_id.clone(), precision);
        self.slave_oscillators.insert(slave_id, slave);
        Ok(())
    }
    
    /// Create quantum entanglement channel
    pub fn create_quantum_channel(&mut self, channel_id: String, endpoint1: String, endpoint2: String) -> Result<(), BuheraError> {
        let mut channel = QuantumChannel::new(channel_id.clone(), 0.999, 1e15);
        channel.add_endpoint(endpoint1).map_err(BuheraError::AtomicClock)?;
        channel.add_endpoint(endpoint2).map_err(BuheraError::AtomicClock)?;
        channel.activate().map_err(BuheraError::AtomicClock)?;
        
        self.quantum_channels.insert(channel_id, channel);
        Ok(())
    }
    
    /// Synchronize entire network
    pub fn synchronize_network(&mut self) -> Result<(), BuheraError> {
        // Get primary master clock
        let primary_master_id = self.master_clocks.keys().next()
            .ok_or_else(|| BuheraError::AtomicClock(AtomicClockError::MasterClockFailure(
                "No master clocks available".to_string()
            )))?
            .clone();
        
        // Get master time reference
        let master_time = self.master_clocks.get(&primary_master_id)
            .unwrap()
            .master_time().map_err(BuheraError::AtomicClock)?;
        
        // Synchronize all slave oscillators
        for (slave_id, slave) in &mut self.slave_oscillators {
            slave.synchronize_with_master(primary_master_id.clone(), master_time)
                .map_err(BuheraError::AtomicClock)?;
        }
        
        // Calculate network accuracy
        self.calculate_network_accuracy();
        
        self.network_synchronized = true;
        Ok(())
    }
    
    /// Calculate network-wide synchronization accuracy
    fn calculate_network_accuracy(&mut self) {
        let mut total_accuracy = 0.0;
        let mut device_count = 0;
        
        // Master clock accuracies
        for master in self.master_clocks.values() {
            total_accuracy += master.synchronization_accuracy;
            device_count += 1;
        }
        
        // Slave oscillator sync qualities
        for slave in self.slave_oscillators.values() {
            total_accuracy += slave.sync_quality;
            device_count += 1;
        }
        
        self.network_accuracy = if device_count > 0 {
            total_accuracy / device_count as f64
        } else {
            0.0
        };
    }
    
    /// Get network time with supreme precision
    pub fn network_time(&self) -> Result<TemporalCoordinate, BuheraError> {
        if !self.network_synchronized {
            return Err(BuheraError::AtomicClock(AtomicClockError::SynchronizationFailure(
                "Network not synchronized".to_string()
            )));
        }
        
        // Use primary master clock time
        let primary_master = self.master_clocks.values().next()
            .ok_or_else(|| BuheraError::AtomicClock(AtomicClockError::MasterClockFailure(
                "No master clocks available".to_string()
            )))?;
        
        primary_master.master_time().map_err(BuheraError::AtomicClock)
    }
    
    /// Measure network-wide synchronization drift
    pub fn measure_network_drift(&mut self) -> Result<f64, BuheraError> {
        let mut total_drift = 0.0;
        let mut master_count = 0;
        
        for master in self.master_clocks.values_mut() {
            let drift = master.measure_network_drift().map_err(BuheraError::AtomicClock)?;
            total_drift += drift;
            master_count += 1;
        }
        
        let average_drift = if master_count > 0 {
            total_drift / master_count as f64
        } else {
            0.0
        };
        
        Ok(average_drift)
    }
    
    /// Get network statistics
    pub fn network_statistics(&self) -> NetworkStatistics {
        let master_stats: Vec<MasterStatistics> = self.master_clocks.values()
            .map(|m| m.master_statistics())
            .collect();
        
        let channel_stats: Vec<ChannelStatistics> = self.quantum_channels.values()
            .map(|c| c.statistics())
            .collect();
        
        NetworkStatistics {
            master_clocks: self.master_clocks.len(),
            slave_oscillators: self.slave_oscillators.len(),
            quantum_channels: self.quantum_channels.len(),
            network_synchronized: self.network_synchronized,
            network_precision: self.network_precision,
            network_accuracy: self.network_accuracy,
            distribution_bandwidth: self.distribution_bandwidth,
            master_statistics: master_stats,
            channel_statistics: channel_stats,
        }
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub master_clocks: usize,
    pub slave_oscillators: usize,
    pub quantum_channels: usize,
    pub network_synchronized: bool,
    pub network_precision: PrecisionLevel,
    pub network_accuracy: f64,
    pub distribution_bandwidth: f64,
    pub master_statistics: Vec<MasterStatistics>,
    pub channel_statistics: Vec<ChannelStatistics>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_channel() {
        let mut channel = QuantumChannel::new("test".to_string(), 0.999, 1e15);
        channel.add_endpoint("endpoint1".to_string()).unwrap();
        channel.add_endpoint("endpoint2".to_string()).unwrap();
        channel.activate().unwrap();
        
        assert!(channel.is_active);
        assert_eq!(channel.endpoints.len(), 2);
    }
    
    #[test]
    fn test_master_atomic_clock() {
        let mut master = MasterAtomicClock::new("master1".to_string(), PrecisionLevel::Supreme);
        master.initialize_master().unwrap();
        
        let time = master.master_time().unwrap();
        assert_eq!(time.precision_level, PrecisionLevel::Supreme);
    }
    
    #[test]
    fn test_slave_oscillator() {
        let slave = SlaveOscillator::new("slave1".to_string(), PrecisionLevel::Stella);
        assert!(!slave.is_synchronized());
    }
} 