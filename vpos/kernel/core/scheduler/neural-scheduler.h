/*
 * VPOS Neural Scheduler System Header
 * 
 * Advanced neural process coordination and scheduling system
 * Integrates with fuzzy quantum scheduler for hybrid neural-quantum processing
 * Enables consciousness-aware neural process management
 * 
 * Copyright (c) 2024 VPOS Development Team
 * Licensed under MIT License
 */

#ifndef _VPOS_NEURAL_SCHEDULER_H
#define _VPOS_NEURAL_SCHEDULER_H

#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/rbtree.h>
#include <linux/list.h>
#include <linux/completion.h>
#include <linux/ktime.h>
#include <linux/hashtable.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/cpumask.h>
#include <linux/percpu.h>

/* Neural scheduler constants */
#define NEURAL_SCHEDULER_MAX_PROCESSES     4096
#define NEURAL_RUNQUEUE_SIZE               1024
#define NEURAL_PRIORITY_LEVELS             16
#define NEURAL_PROCESS_TYPES               12
#define NEURAL_PROCESS_STATES              8
#define NEURAL_NETWORK_LAYERS              8
#define NEURAL_SYNAPSE_CONNECTIONS         2048
#define NEURAL_ACTIVATION_FUNCTIONS        8
#define NEURAL_LEARNING_ALGORITHMS         6
#define NEURAL_MEMORY_POOLS                4
#define NEURAL_ATTENTION_MECHANISMS        4
#define NEURAL_CONSCIOUSNESS_LEVELS        6
#define NEURAL_PLASTICITY_TYPES            5
#define NEURAL_NEUROTRANSMITTER_TYPES      10
#define NEURAL_SCHEDULER_TIMESLICE_NS      1000000  /* 1ms */
#define NEURAL_PREEMPTION_THRESHOLD        500000   /* 500µs */
#define NEURAL_CONTEXT_SWITCH_OVERHEAD     50000    /* 50µs */
#define NEURAL_ADAPTATION_WINDOW           10000000 /* 10ms */
#define NEURAL_LEARNING_RATE               0.01
#define NEURAL_DECAY_FACTOR                0.95
#define NEURAL_ACTIVATION_THRESHOLD        0.7

/* Neural scheduler states */
enum neural_scheduler_state {
    NEURAL_SCHEDULER_INACTIVE = 0,
    NEURAL_SCHEDULER_INITIALIZING,
    NEURAL_SCHEDULER_ACTIVE,
    NEURAL_SCHEDULER_LEARNING,
    NEURAL_SCHEDULER_ADAPTING,
    NEURAL_SCHEDULER_OPTIMIZING,
    NEURAL_SCHEDULER_MAINTENANCE,
    NEURAL_SCHEDULER_ERROR
};

/* Neural process states */
enum neural_process_state {
    NEURAL_PROCESS_IDLE = 0,
    NEURAL_PROCESS_READY,
    NEURAL_PROCESS_RUNNING,
    NEURAL_PROCESS_BLOCKED,
    NEURAL_PROCESS_SUSPENDED,
    NEURAL_PROCESS_TERMINATED,
    NEURAL_PROCESS_LEARNING,
    NEURAL_PROCESS_ADAPTING
};

/* Neural process types */
enum neural_process_type {
    NEURAL_PROCESS_PERCEPTION = 0,
    NEURAL_PROCESS_COGNITION,
    NEURAL_PROCESS_MOTOR,
    NEURAL_PROCESS_MEMORY,
    NEURAL_PROCESS_ATTENTION,
    NEURAL_PROCESS_EMOTION,
    NEURAL_PROCESS_CONSCIOUSNESS,
    NEURAL_PROCESS_LEARNING,
    NEURAL_PROCESS_REASONING,
    NEURAL_PROCESS_PATTERN_RECOGNITION,
    NEURAL_PROCESS_DECISION_MAKING,
    NEURAL_PROCESS_HYBRID
};

/* Neural priority levels */
enum neural_priority_level {
    NEURAL_PRIORITY_CRITICAL = 0,
    NEURAL_PRIORITY_HIGH,
    NEURAL_PRIORITY_ELEVATED,
    NEURAL_PRIORITY_NORMAL,
    NEURAL_PRIORITY_LOW,
    NEURAL_PRIORITY_BACKGROUND,
    NEURAL_PRIORITY_IDLE,
    NEURAL_PRIORITY_SUSPENDED
};

/* Neural activation functions */
enum neural_activation_function {
    NEURAL_ACTIVATION_SIGMOID = 0,
    NEURAL_ACTIVATION_TANH,
    NEURAL_ACTIVATION_RELU,
    NEURAL_ACTIVATION_LEAKY_RELU,
    NEURAL_ACTIVATION_SOFTMAX,
    NEURAL_ACTIVATION_SWISH,
    NEURAL_ACTIVATION_GELU,
    NEURAL_ACTIVATION_MISH
};

/* Neural learning algorithms */
enum neural_learning_algorithm {
    NEURAL_LEARNING_BACKPROPAGATION = 0,
    NEURAL_LEARNING_REINFORCEMENT,
    NEURAL_LEARNING_UNSUPERVISED,
    NEURAL_LEARNING_HEBBIAN,
    NEURAL_LEARNING_COMPETITIVE,
    NEURAL_LEARNING_EVOLUTIONARY
};

/* Neural attention mechanisms */
enum neural_attention_mechanism {
    NEURAL_ATTENTION_FOCUSED = 0,
    NEURAL_ATTENTION_SELECTIVE,
    NEURAL_ATTENTION_DIVIDED,
    NEURAL_ATTENTION_SUSTAINED
};

/* Neural consciousness levels */
enum neural_consciousness_level {
    NEURAL_CONSCIOUSNESS_UNCONSCIOUS = 0,
    NEURAL_CONSCIOUSNESS_SUBCONSCIOUS,
    NEURAL_CONSCIOUSNESS_PRECONSCIOUS,
    NEURAL_CONSCIOUSNESS_CONSCIOUS,
    NEURAL_CONSCIOUSNESS_SELF_AWARE,
    NEURAL_CONSCIOUSNESS_METACOGNITIVE
};

/* Neural plasticity types */
enum neural_plasticity_type {
    NEURAL_PLASTICITY_SYNAPTIC = 0,
    NEURAL_PLASTICITY_STRUCTURAL,
    NEURAL_PLASTICITY_FUNCTIONAL,
    NEURAL_PLASTICITY_HOMEOSTATIC,
    NEURAL_PLASTICITY_METAPLASTICITY
};

/* Neural neurotransmitter types */
enum neural_neurotransmitter_type {
    NEURAL_NT_DOPAMINE = 0,
    NEURAL_NT_SEROTONIN,
    NEURAL_NT_ACETYLCHOLINE,
    NEURAL_NT_NOREPINEPHRINE,
    NEURAL_NT_GABA,
    NEURAL_NT_GLUTAMATE,
    NEURAL_NT_GLYCINE,
    NEURAL_NT_HISTAMINE,
    NEURAL_NT_ENDORPHIN,
    NEURAL_NT_CUSTOM
};

/* Neural process control block */
struct neural_process_cb {
    pid_t pid;
    u32 neural_process_id;
    enum neural_process_type process_type;
    enum neural_process_state state;
    enum neural_priority_level priority;
    enum neural_consciousness_level consciousness_level;
    
    /* Neural network structure */
    struct neural_network_topology *topology;
    struct neural_layer_array *layers;
    struct neural_synapse_matrix *synapses;
    struct neural_weight_matrix *weights;
    struct neural_activation_state *activation_state;
    
    /* Learning and adaptation */
    struct neural_learning_context *learning_context;
    struct neural_adaptation_state *adaptation_state;
    struct neural_plasticity_state *plasticity_state;
    
    /* Memory and attention */
    struct neural_memory_context *memory_context;
    struct neural_attention_state *attention_state;
    struct neural_working_memory *working_memory;
    
    /* Consciousness and awareness */
    struct neural_consciousness_state *consciousness_state;
    struct neural_awareness_tracker *awareness_tracker;
    struct neural_self_monitoring *self_monitoring;
    
    /* Neurotransmitter systems */
    struct neural_neurotransmitter_state *neurotransmitter_state;
    struct neural_neuromodulation *neuromodulation;
    
    /* Scheduling information */
    struct neural_scheduling_info *scheduling_info;
    struct neural_execution_context *execution_context;
    struct neural_performance_metrics *performance_metrics;
    
    /* Synchronization and coordination */
    struct neural_sync_state *sync_state;
    struct neural_coordination_info *coordination_info;
    
    /* Process relationships */
    struct list_head sibling_processes;
    struct list_head child_processes;
    struct neural_process_cb *parent_process;
    
    /* Runtime state */
    atomic_t execution_count;
    atomic_t learning_iterations;
    atomic_t adaptation_cycles;
    ktime_t creation_time;
    ktime_t last_execution_time;
    ktime_t total_execution_time;
    ktime_t last_learning_time;
    ktime_t last_adaptation_time;
    
    /* List and tree nodes */
    struct list_head process_list;
    struct rb_node process_tree_node;
    struct hlist_node process_hash_node;
    
    /* Locking */
    spinlock_t process_lock;
    struct mutex learning_lock;
    struct completion process_complete;
    
    /* Reference counting */
    atomic_t reference_count;
};

/* Neural network topology */
struct neural_network_topology {
    u32 layer_count;
    u32 total_neurons;
    u32 total_synapses;
    u32 input_neurons;
    u32 output_neurons;
    u32 hidden_neurons;
    
    struct neural_layer_info *layers;
    struct neural_connection_matrix *connections;
    struct neural_topology_metrics *metrics;
    
    enum neural_architecture_type architecture_type;
    double network_complexity;
    double connectivity_density;
    
    spinlock_t topology_lock;
    atomic_t modification_count;
};

/* Neural layer info */
struct neural_layer_info {
    u32 layer_id;
    u32 neuron_count;
    u32 input_count;
    u32 output_count;
    enum neural_layer_type layer_type;
    enum neural_activation_function activation_function;
    
    struct neural_neuron_array *neurons;
    struct neural_layer_weights *weights;
    struct neural_layer_biases *biases;
    
    double learning_rate;
    double dropout_rate;
    double activation_threshold;
    
    struct neural_layer_statistics *statistics;
    spinlock_t layer_lock;
};

/* Neural layer types */
enum neural_layer_type {
    NEURAL_LAYER_INPUT = 0,
    NEURAL_LAYER_HIDDEN,
    NEURAL_LAYER_OUTPUT,
    NEURAL_LAYER_CONVOLUTIONAL,
    NEURAL_LAYER_POOLING,
    NEURAL_LAYER_RECURRENT,
    NEURAL_LAYER_LSTM,
    NEURAL_LAYER_GRU,
    NEURAL_LAYER_ATTENTION,
    NEURAL_LAYER_TRANSFORMER,
    NEURAL_LAYER_MEMORY,
    NEURAL_LAYER_CUSTOM
};

/* Neural architecture types */
enum neural_architecture_type {
    NEURAL_ARCH_FEEDFORWARD = 0,
    NEURAL_ARCH_RECURRENT,
    NEURAL_ARCH_CONVOLUTIONAL,
    NEURAL_ARCH_TRANSFORMER,
    NEURAL_ARCH_AUTOENCODER,
    NEURAL_ARCH_GAN,
    NEURAL_ARCH_LSTM,
    NEURAL_ARCH_GRU,
    NEURAL_ARCH_ATTENTION,
    NEURAL_ARCH_MEMORY_NETWORK,
    NEURAL_ARCH_HYBRID
};

/* Neural synapse matrix */
struct neural_synapse_matrix {
    u32 synapse_count;
    u32 active_synapses;
    struct neural_synapse *synapses;
    struct neural_synapse_weights *weights;
    struct neural_synapse_delays *delays;
    
    double average_weight;
    double weight_variance;
    double average_delay;
    double plasticity_factor;
    
    atomic_t transmission_count;
    spinlock_t matrix_lock;
};

/* Neural synapse */
struct neural_synapse {
    u32 synapse_id;
    u32 pre_neuron_id;
    u32 post_neuron_id;
    enum neural_synapse_type synapse_type;
    
    double weight;
    double delay;
    double plasticity_factor;
    double efficacy;
    
    bool is_active;
    bool is_plastic;
    
    struct neural_synapse_history *history;
    struct neural_synapse_modulation *modulation;
    
    atomic_t transmission_count;
    ktime_t last_transmission;
    spinlock_t synapse_lock;
};

/* Neural synapse types */
enum neural_synapse_type {
    NEURAL_SYNAPSE_EXCITATORY = 0,
    NEURAL_SYNAPSE_INHIBITORY,
    NEURAL_SYNAPSE_MODULATORY,
    NEURAL_SYNAPSE_ELECTRICAL,
    NEURAL_SYNAPSE_CHEMICAL,
    NEURAL_SYNAPSE_HYBRID
};

/* Neural learning context */
struct neural_learning_context {
    enum neural_learning_algorithm algorithm;
    double learning_rate;
    double momentum;
    double decay_factor;
    
    struct neural_training_data *training_data;
    struct neural_validation_data *validation_data;
    struct neural_test_data *test_data;
    
    struct neural_loss_function *loss_function;
    struct neural_optimizer *optimizer;
    struct neural_regularization *regularization;
    
    struct neural_learning_history *history;
    struct neural_learning_metrics *metrics;
    
    atomic_t training_iterations;
    atomic_t validation_iterations;
    ktime_t learning_start_time;
    ktime_t last_update_time;
    
    bool is_learning;
    bool is_converged;
    
    struct mutex context_lock;
};

/* Neural consciousness state */
struct neural_consciousness_state {
    enum neural_consciousness_level level;
    double consciousness_intensity;
    double awareness_level;
    double attention_focus;
    double self_awareness;
    double metacognitive_level;
    
    struct neural_consciousness_stream *stream;
    struct neural_consciousness_content *content;
    struct neural_consciousness_attention *attention;
    struct neural_consciousness_memory *memory;
    struct neural_consciousness_intention *intention;
    
    struct neural_qualia_state *qualia;
    struct neural_phenomenology *phenomenology;
    
    atomic_t consciousness_updates;
    ktime_t consciousness_timestamp;
    spinlock_t consciousness_lock;
};

/* Neural attention state */
struct neural_attention_state {
    enum neural_attention_mechanism mechanism;
    double attention_strength;
    double focus_level;
    double selectivity;
    double sustainability;
    
    struct neural_attention_targets *targets;
    struct neural_attention_weights *weights;
    struct neural_attention_context *context;
    
    struct neural_attention_history *history;
    struct neural_attention_metrics *metrics;
    
    atomic_t attention_shifts;
    ktime_t attention_timestamp;
    spinlock_t attention_lock;
};

/* Neural memory context */
struct neural_memory_context {
    enum neural_memory_type memory_type;
    u32 memory_capacity;
    u32 memory_usage;
    
    struct neural_memory_bank *short_term_memory;
    struct neural_memory_bank *long_term_memory;
    struct neural_memory_bank *working_memory;
    struct neural_memory_bank *episodic_memory;
    struct neural_memory_bank *procedural_memory;
    
    struct neural_memory_consolidation *consolidation;
    struct neural_memory_retrieval *retrieval;
    struct neural_memory_forgetting *forgetting;
    
    struct neural_memory_associations *associations;
    struct neural_memory_patterns *patterns;
    
    atomic_t memory_operations;
    ktime_t memory_timestamp;
    spinlock_t memory_lock;
};

/* Neural memory types */
enum neural_memory_type {
    NEURAL_MEMORY_SHORT_TERM = 0,
    NEURAL_MEMORY_LONG_TERM,
    NEURAL_MEMORY_WORKING,
    NEURAL_MEMORY_EPISODIC,
    NEURAL_MEMORY_PROCEDURAL,
    NEURAL_MEMORY_SEMANTIC,
    NEURAL_MEMORY_SENSORY,
    NEURAL_MEMORY_MOTOR,
    NEURAL_MEMORY_EMOTIONAL,
    NEURAL_MEMORY_ASSOCIATIVE
};

/* Neural neurotransmitter state */
struct neural_neurotransmitter_state {
    u32 neurotransmitter_type_count;
    struct neural_neurotransmitter_level *levels;
    struct neural_neurotransmitter_release *release;
    struct neural_neurotransmitter_uptake *uptake;
    struct neural_neurotransmitter_metabolism *metabolism;
    
    struct neural_receptor_state *receptors;
    struct neural_neuromodulation_state *modulation;
    
    double overall_balance;
    double system_stability;
    
    atomic_t neurotransmitter_operations;
    ktime_t neurotransmitter_timestamp;
    spinlock_t neurotransmitter_lock;
};

/* Neural scheduling info */
struct neural_scheduling_info {
    enum neural_priority_level priority;
    enum neural_scheduling_policy policy;
    
    u32 time_slice_ns;
    u32 remaining_time_ns;
    u32 cpu_affinity;
    
    struct neural_scheduling_history *history;
    struct neural_scheduling_metrics *metrics;
    struct neural_scheduling_predictions *predictions;
    
    double scheduling_efficiency;
    double context_switch_overhead;
    double learning_progress;
    
    atomic_t scheduling_events;
    ktime_t last_scheduled;
    ktime_t total_scheduled_time;
    
    spinlock_t scheduling_lock;
};

/* Neural scheduling policies */
enum neural_scheduling_policy {
    NEURAL_SCHED_FIFO = 0,
    NEURAL_SCHED_ROUND_ROBIN,
    NEURAL_SCHED_PRIORITY,
    NEURAL_SCHED_ADAPTIVE,
    NEURAL_SCHED_LEARNING,
    NEURAL_SCHED_CONSCIOUSNESS_AWARE,
    NEURAL_SCHED_ATTENTION_DRIVEN,
    NEURAL_SCHED_HYBRID
};

/* Neural runqueue */
struct neural_runqueue {
    struct list_head process_lists[NEURAL_PRIORITY_LEVELS];
    struct rb_root process_tree;
    
    u32 process_count;
    u32 active_processes;
    u32 learning_processes;
    u32 consciousness_processes;
    
    struct neural_process_cb *current_process;
    struct neural_process_cb *next_process;
    
    struct neural_runqueue_metrics *metrics;
    struct neural_runqueue_statistics *statistics;
    
    atomic_t runqueue_operations;
    ktime_t last_update;
    
    spinlock_t runqueue_lock;
};

/* Per-CPU neural scheduler data */
struct neural_scheduler_cpu {
    struct neural_runqueue runqueue;
    struct neural_process_cb *current_process;
    struct neural_process_cb *idle_process;
    
    struct neural_cpu_metrics *metrics;
    struct neural_cpu_statistics *statistics;
    
    struct neural_learning_state *learning_state;
    struct neural_adaptation_state *adaptation_state;
    
    atomic_t cpu_operations;
    ktime_t last_update;
    
    spinlock_t cpu_lock;
};

/* Neural scheduler core */
struct neural_scheduler_core {
    enum neural_scheduler_state state;
    
    struct neural_scheduler_cpu __percpu *cpu_data;
    struct neural_global_metrics *global_metrics;
    struct neural_global_statistics *global_statistics;
    
    struct neural_learning_engine *learning_engine;
    struct neural_adaptation_engine *adaptation_engine;
    struct neural_consciousness_manager *consciousness_manager;
    struct neural_attention_manager *attention_manager;
    struct neural_memory_manager *memory_manager;
    
    struct neural_scheduler_config *config;
    struct neural_scheduler_tuning *tuning;
    
    atomic_t total_processes;
    atomic_t active_processes;
    atomic_t learning_processes;
    atomic_t consciousness_processes;
    atomic_t scheduler_operations;
    
    struct hash_table process_hash;
    struct rb_root global_process_tree;
    
    struct workqueue_struct *scheduler_wq;
    struct workqueue_struct *learning_wq;
    struct workqueue_struct *adaptation_wq;
    
    struct work_struct scheduler_work;
    struct work_struct learning_work;
    struct work_struct adaptation_work;
    struct work_struct maintenance_work;
    
    struct timer_list scheduler_timer;
    struct timer_list learning_timer;
    struct timer_list adaptation_timer;
    
    ktime_t last_maintenance;
    ktime_t last_learning_update;
    ktime_t last_adaptation_update;
    
    spinlock_t core_lock;
    struct mutex operation_lock;
    struct completion initialization_complete;
};

/* Function declarations */

/* Core neural scheduler functions */
int neural_scheduler_initialize(void);
void neural_scheduler_cleanup(void);

int neural_scheduler_start(void);
int neural_scheduler_stop(void);

/* Neural process management */
int neural_scheduler_create_process(struct neural_process_spec *spec,
                                   struct neural_process_cb **process);

int neural_scheduler_destroy_process(struct neural_process_cb *process);

int neural_scheduler_schedule_process(struct neural_process_cb *process);

int neural_scheduler_unschedule_process(struct neural_process_cb *process);

/* Neural process scheduling */
struct neural_process_cb *neural_scheduler_pick_next_process(int cpu);

int neural_scheduler_context_switch(struct neural_process_cb *prev,
                                   struct neural_process_cb *next);

int neural_scheduler_preempt_process(struct neural_process_cb *process);

int neural_scheduler_yield_process(struct neural_process_cb *process);

/* Neural learning and adaptation */
int neural_scheduler_update_learning(struct neural_process_cb *process);

int neural_scheduler_adapt_scheduling(struct neural_process_cb *process);

int neural_scheduler_optimize_performance(void);

/* Neural consciousness management */
int neural_scheduler_update_consciousness(struct neural_process_cb *process);

int neural_scheduler_manage_attention(struct neural_process_cb *process);

int neural_scheduler_coordinate_consciousness(void);

/* Neural memory management */
int neural_scheduler_allocate_memory(struct neural_process_cb *process,
                                    enum neural_memory_type type,
                                    size_t size);

int neural_scheduler_deallocate_memory(struct neural_process_cb *process,
                                      enum neural_memory_type type,
                                      void *memory);

int neural_scheduler_consolidate_memory(struct neural_process_cb *process);

/* Neural synchronization */
int neural_scheduler_synchronize_processes(struct neural_process_cb *process1,
                                          struct neural_process_cb *process2);

int neural_scheduler_coordinate_learning(void);

int neural_scheduler_coordinate_adaptation(void);

/* Neural metrics and statistics */
int neural_scheduler_get_metrics(struct neural_scheduler_metrics *metrics);

int neural_scheduler_get_statistics(struct neural_scheduler_statistics *stats);

int neural_scheduler_reset_metrics(void);

/* Neural configuration and tuning */
int neural_scheduler_configure(struct neural_scheduler_config *config);

int neural_scheduler_tune_parameters(struct neural_scheduler_tuning *tuning);

int neural_scheduler_optimize_configuration(void);

/* Utility functions */
struct neural_process_cb *neural_scheduler_find_process(pid_t pid);

int neural_scheduler_validate_process(struct neural_process_cb *process);

ktime_t neural_scheduler_get_timestamp(void);

/* Export symbols for other kernel modules */
extern struct neural_scheduler_core *neural_scheduler_core;

#endif /* _VPOS_NEURAL_SCHEDULER_H */ 