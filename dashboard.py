# -*- coding: utf-8 -*-
"""
Wordle Game Prediction Dashboard
Interactive dashboard for Wordle game prediction model results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "07_results")
ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "06_analysis", "output")
ATTENTION_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "08_attention_analysis", "output")

# =====================
# Load data
# =====================
def load_model_results():
    """
    Load model evaluation results
    """
    results_file = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")
    if not os.path.exists(results_file):
        st.error(f"Model evaluation results file not found at {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    return df

# =====================
# Dashboard layout
# =====================

def main():
    """
    Main function to create the dashboard
    """
    # Set page configuration
    st.set_page_config(
        page_title="Wordle Game Prediction Dashboard",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load data
    model_results_df = load_model_results()
    
    if model_results_df is None:
        st.stop()
    
    # Title and introduction
    st.title("🎮 Wordle Game Prediction Dashboard")
    st.markdown("---")
    
    st.markdown("""
    This dashboard displays the results of Wordle game prediction models. 
    You can select different models and parameters to view their performance.
    """)
    
    # Sidebar for parameter selection
    st.sidebar.header("🔧 Parameter Selection")
    
    # Model selection - allow multiple models
    model_names = model_results_df['model'].tolist()
    selected_models = st.sidebar.multiselect(
        "Select Model(s)",
        model_names,
        default=model_names
    )
    
    # Metric selection
    metrics = ['mae', 'rmse', 'accuracy', 'f1_score', 'auc']
    metric_names = {
        'mae': 'MAE',
        'rmse': 'RMSE',
        'accuracy': 'Accuracy',
        'f1_score': 'F1-Score',
        'auc': 'AUC'
    }
    
    selected_metric = st.sidebar.selectbox(
        "Select Metric",
        metrics,
        format_func=lambda x: metric_names[x],
        index=2  # Default to accuracy
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Model Performance Comparison")
        
        # Filter data based on selected models
        filtered_df = model_results_df[model_results_df['model'].isin(selected_models)]
        
        # Create a bar chart for selected metric
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort models by selected metric
        if selected_metric in ['mae', 'rmse']:
            # Lower is better for MAE and RMSE
            sorted_df = filtered_df.sort_values(by=selected_metric, ascending=True)
        else:
            # Higher is better for accuracy, f1_score, and auc
            sorted_df = filtered_df.sort_values(by=selected_metric, ascending=False)
        
        # Define colors for each model
        model_colors = {
            'lstm': 'skyblue',
            'lstm_attention': 'lightgreen',
            'bilstm': 'salmon',
            'transformer': 'gold'
        }
        colors = [model_colors[model] for model in sorted_df['model']]
        
        # Create bar chart
        bars = ax.bar(
            sorted_df['model'],
            sorted_df[selected_metric],
            color=colors
        )
        
        # Add value labels on top of bars
        for bar, value in zip(bars, sorted_df[selected_metric]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + height*0.01,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_names[selected_metric])
        ax.set_title(f'{metric_names[selected_metric]} Comparison Across Models')
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
    
    with col2:
        st.header("📋 Model Details")
        
        # Display details for multiple selected models
        if len(selected_models) == 1:
            # Single model selected
            selected_model = selected_models[0]
            selected_model_df = model_results_df[model_results_df['model'] == selected_model]
            
            if not selected_model_df.empty:
                model_data = selected_model_df.iloc[0].to_dict()
                
                st.subheader(f"{selected_model} Model")
                
                # Display metrics in a table
                metrics_table = pd.DataFrame({
                    'Metric': [metric_names[m] for m in metrics],
                    'Value': [model_data[m] if m in model_data and not pd.isna(model_data[m]) else 'N/A' for m in metrics]
                })
                
                st.table(metrics_table)
        else:
            # Multiple models selected - display comparison table
            st.subheader("📊 Selected Models Comparison")
            
            # Create comparison table
            comparison_df = filtered_df.copy()
            comparison_df = comparison_df[['model'] + metrics]
            
            # Rename columns to be more user-friendly
            comparison_df.columns = ['Model'] + [metric_names[m] for m in metrics]
            
            # Sort by selected metric
            if selected_metric in ['mae', 'rmse']:
                # Lower is better
                comparison_df = comparison_df.sort_values(by=metric_names[selected_metric], ascending=True)
            else:
                # Higher is better
                comparison_df = comparison_df.sort_values(by=metric_names[selected_metric], ascending=False)
            
            st.dataframe(comparison_df.style.format(precision=4))
        
        st.markdown("---")
        
        st.header("🏆 Best Model by Metric")
        
        # Display best model for each metric
        best_models = {}
        for metric in metrics:
            if metric in ['mae', 'rmse']:
                # Lower is better
                best_idx = model_results_df[metric].idxmin()
            else:
                # Higher is better
                best_idx = model_results_df[metric].idxmax()
            best_models[metric] = model_results_df.loc[best_idx, 'model']
        
        best_models_df = pd.DataFrame({
            'Metric': [metric_names[m] for m in best_models.keys()],
            'Best Model': list(best_models.values())
        })
        
        st.table(best_models_df)
    
    # Add model-specific visualizations
    st.header("🖼️ Model-Specific Visualizations")
    
    # Single model selection for detailed visualizations
    selected_single_model = st.selectbox(
        "Select a Model for Detailed Visualizations",
        model_names,
        index=0
    )
    
    # Get model output directory
    MODEL_DIR = os.path.join(PROJECT_ROOT, "05_models", selected_single_model)
    MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "output")
    
    # Define the four chart types to display
    chart_types = [
        f"{selected_single_model}_training_curves.png",
        f"{selected_single_model}_metrics_summary.png",
        f"{selected_single_model}_prediction_analysis.png",
        f"{selected_single_model}_error_by_activity.png"
    ]
    
    # Display the four charts in a 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        # First chart - Training Curves
        chart_path = os.path.join(MODEL_OUTPUT_DIR, chart_types[0])
        if os.path.exists(chart_path):
            st.subheader(f"📉 {selected_single_model} - Training Curves")
            st.image(chart_path, width='stretch')
        else:
            st.warning(f"Chart not found: {chart_path}")
        
        # Second chart - Metrics Summary
        chart_path = os.path.join(MODEL_OUTPUT_DIR, chart_types[1])
        if os.path.exists(chart_path):
            st.subheader(f"📊 {selected_single_model} - Metrics Summary")
            st.image(chart_path, width='stretch')
        else:
            st.warning(f"Chart not found: {chart_path}")
    
    with col2:
        # Third chart - Prediction Analysis
        chart_path = os.path.join(MODEL_OUTPUT_DIR, chart_types[2])
        if os.path.exists(chart_path):
            st.subheader(f"🔍 {selected_single_model} - Prediction Analysis")
            st.image(chart_path, width='stretch')
        else:
            st.warning(f"Chart not found: {chart_path}")
        
        # Fourth chart - Error by Activity
        chart_path = os.path.join(MODEL_OUTPUT_DIR, chart_types[3])
        if os.path.exists(chart_path):
            st.subheader(f"❌ {selected_single_model} - Error by Activity")
            st.image(chart_path, width='stretch')
        else:
            st.warning(f"Chart not found: {chart_path}")
    
    st.markdown("---")
    
    # Display specified comparison charts
    st.header("🔄 Model Comparison Visualizations")
    
    # Display all metrics comparison chart
    all_metrics_path = os.path.join(ANALYSIS_OUTPUT_DIR, "model_comparison_all_metrics.png")
    if os.path.exists(all_metrics_path):
        st.subheader("📊 All Metrics Comparison")
        st.image(all_metrics_path, width='stretch')
    else:
        st.warning(f"Chart not found: {all_metrics_path}")
    
    # Display radar chart comparison
    radar_path = os.path.join(ANALYSIS_OUTPUT_DIR, "model_comparison_radar.png")
    if os.path.exists(radar_path):
        st.subheader("📈 Radar Chart Comparison")
        st.image(radar_path, width='stretch')
    else:
        st.warning(f"Chart not found: {radar_path}")
    
    st.markdown("---")
    
    # Add more insights and information
    st.header("🔍 Model Insights")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🧠 Model Architectures")
        st.markdown("""
        | Model | Architecture | Key Features |
        |-------|--------------|--------------|
        | LSTM | Long Short-Term Memory | Sequential data modeling, long-term dependencies |
        | LSTM + Attention | LSTM with Attention Mechanism | Focuses on relevant parts of input sequence |
        | BiLSTM | Bidirectional LSTM | Processes sequence in both forward and backward directions |
        | Transformer | Transformer with Self-Attention | Parallel processing, long-range dependencies |
        """)
    
    with col_b:
        st.subheader("📈 Training Information")
        st.markdown("""
        - **Batch Size**: 128
        - **Epochs**: 30 with early stopping
        - **Optimizer**: Adam
        - **Loss Functions**: 
          - Regression: Mean Absolute Error (MAE)
          - Classification: Binary Crossentropy
        """)
    
    # Attention Mechanism Analysis
    st.markdown("---")
    st.header("🧠 Attention Mechanism Analysis")
    
    # Display attention weights bar chart
    attention_bar_path = os.path.join(ATTENTION_ANALYSIS_DIR, "attention_weights_bar.png")
    if os.path.exists(attention_bar_path):
        st.subheader("📊 Attention Weights Distribution")
        st.image(attention_bar_path, width='stretch')
        
        # Add explanation text
        st.markdown("""
        **Attention Weights Interpretation**:
        - This chart displays the average attention weights across different sequence steps (historical attempts)
        - **Step 3** exhibits the highest attention weight (approximately 0.2009), significantly higher than the average weight of around 0.13 for other steps
        - Attention weights indicate the degree of focus the model assigns to each historical attempt when making predictions
        - This visualization enhances model interpretability by revealing which steps are most critical for decision-making
        
        **Key Insights**:
        - The model prioritizes the middle attempt (Step 3), suggesting it contains the most valuable information about the player's progress toward solving the Wordle puzzle
        - In the Wordle game mechanism, the first two attempts provide limited information as players explore the word space, while Step 3 typically occurs after eliminating some incorrect letters and identifying some correct ones
        - This middle step's feedback plays a crucial role in narrowing down the search space and clarifying the guessing direction
        - Other steps maintain relatively balanced attention weights, indicating the model still considers the complete sequence information while emphasizing the most critical step
        """)
    else:
        st.warning(f"Attention weights bar chart not found at {attention_bar_path}")
    
    # Conclusion section
    st.markdown("---")
    st.header("🔍 Conclusion")
    
    # Find the best model overall
    best_auc_model = model_results_df.loc[model_results_df['auc'].idxmax()]['model']
    best_accuracy_model = model_results_df.loc[model_results_df['accuracy'].idxmax()]['model']
    best_mae_model = model_results_df.loc[model_results_df['mae'].idxmin()]['model']
    
    st.markdown(f"""
    Based on the comprehensive analysis of Wordle game prediction models using over 6.8 million real player game records:
    
    - **Best AUC Performance**: **{best_auc_model}** with an AUC of {model_results_df['auc'].max():.4f}
    - **Best Accuracy**: **{best_accuracy_model}** with an accuracy of {model_results_df['accuracy'].max():.4f}
    - **Best MAE**: **{best_mae_model}** with an MAE of {model_results_df['mae'].min():.4f}
    
    ## Key Findings
    
    1. **Effective Sequence Modeling**: All models (LSTM, BiLSTM, LSTM-Attention, Transformer) achieve stable and consistent prediction performance in both regression and classification tasks, demonstrating that player behavior follows learnable patterns.
    
    2. **Bidirectional Information Advantage**: The BiLSTM model performs best in regression error metrics (MAE, RMSE) and classification discrimination (AUC), showing that leveraging both forward and backward sequence context enhances the model's understanding of player behavior.
    
    3. **Attention Mechanism Insights**: The attention weight analysis reveals that the model assigns highest importance to the middle attempt (Step 3), which aligns with the Wordle game's "information rapid convergence stage" in actual cognitive processes.
    
    4. **Complexity vs Performance**: In this short-sequence, low-dimensional feature scenario, the Transformer model does not significantly outperform models based on cyclic structures, highlighting that matching model structure to data characteristics is more important than simply increasing complexity.
    
    ## Model Selection Recommendations
    
    - **Overall Best Performance**: BiLSTM model, which offers balanced performance in both regression and classification tasks
    - **Classification Precision Focus**: LSTM-Attention model, where the attention mechanism enhances the focus on key steps
    - **Cost-Effective Option**: Basic LSTM model, which provides stable performance with lower complexity
    - **Not Recommended**: Transformer model, considering its higher computational cost without significant performance advantages
    """)
    
    # Recommendations and Future Work
    st.header("💡 Recommendations and Future Work")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Study Advantages")
        st.markdown("""
        1. **Large and Real Data**: Based on millions of real player game records, avoiding biases from small or simulated data
        2. **Reasonable Problem Modeling**: Splitting the Wordle prediction problem into complementary regression and classification tasks
        3. **Systematic and Fair Model Comparison**: All models share identical input features, training strategies, and evaluation metrics
        4. **In-depth Result Analysis**: Beyond metrics, analyzing error distributions across player activity levels and attention weight visualizations
        5. **Complete Engineering Pipeline**: Modular implementation with clear reproducibility and extensibility
        """)
    
    with col2:
        st.subheader("🔍 Limitations and Improvement Directions")
        st.markdown("""
        1. **Data Source Bias**: Data only from Twitter, potentially overrepresenting active or successful players
        2. **Limited Long-term Learning Modeling**: Focusing on single-game sequences, not explicit modeling of player evolution across games
        3. **Class Imbalance Challenge**: Success samples significantly outnumber failure samples, affecting classification performance
        4. **Generic Model Architectures**: Using general sequence models without domain-specific design for Wordle rules
        
        **Future Work**:
        - Incorporate more detailed player behavior features
        - Model cross-game temporal dependencies
        - Address class imbalance through sampling or loss function adjustments
        - Design domain-specific model structures (e.g., letter-level constraint modeling)
        - Explore reinforcement learning and graph structure modeling for human decision processes
        """)
    
    st.subheader("🚀 Practical Recommendations")
    st.markdown(f"""
    - **For research purposes**: Focus on model explainability and player behavior insights
    - **For production deployment**: Prioritize model efficiency and scalability
    - **For game analytics**: Combine model predictions with player segmentation for targeted insights
    - **For user experience optimization**: Use prediction results to design adaptive difficulty levels
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Wordle Game Prediction Dashboard | Created with Streamlit")

# =====================
# Run the dashboard
# =====================
if __name__ == "__main__":
    main()