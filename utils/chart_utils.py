import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.figure import Figure
import io

class ChartConfig:
    """Centralized chart configuration class"""
    
    # Global chart styles
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    


    # Balanced configuration for optimal visual appeal
    SIZE_CONFIG = {
        SMALL: {
            "figsize": {"pie": (4, 4), "bar": (5, 3.5), "line": (5, 3.5)},
            "title_size": 10,     # Readable title
            "label_size": 8,      # Clear axis labels
            "tick_size": 7,       # Legible tick marks
            "legend_size": 7,     # Compact but readable legend
            "annotation_size": 7, # Visible annotations
            "dpi": 100            # Standard resolution
        },
        MEDIUM: {
            "figsize": {"pie": (3, 3), "bar": (6, 3.5), "line": (6, 3.5)},
            "title_size": 12,
            "label_size": 8,
            "tick_size": 7,
            "legend_size": 5,
            "annotation_size": 7,
            "dpi": 200
        },
        LARGE: {
            "figsize": {"pie": (8, 8), "bar": (10, 6), "line": (10, 6)},
            "title_size": 14,
            "label_size": 12,
            "tick_size": 10,
            "legend_size": 10,
            "annotation_size": 10,
            "dpi": 100
        }
    }

# In utils/chart_utils.py


    @classmethod
    def create_pie_chart(cls, data, labels, title, size=MEDIUM, min_percent_for_label=5.0):
        """
        Create a standardized pie chart with optimal visual appeal
        
        Args:
            data: Values for pie chart segments
            labels: Labels for each segment
            title: Chart title
            size: Chart size (SMALL, MEDIUM, LARGE)
            min_percent_for_label: Minimum percentage to show direct label
            
        Returns:
            Figure: Matplotlib figure
        """
        config = cls.SIZE_CONFIG.get(size, cls.SIZE_CONFIG[cls.MEDIUM])
        
        # Create figure
        fig, ax = plt.subplots(figsize=config["figsize"]["pie"], dpi=config["dpi"])
        
        # Calculate percentages
        total = sum(data)
        percentages = [(val/total * 100) for val in data]
        
        # Function to conditionally show percentages
        def make_autopct(pcts):
            def autopct(pct):
                return f'{pct:.1f}%' if pct >= min_percent_for_label else ''
            return autopct
        
        # Create pie chart with conditional percentage labels
        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        wedges, texts, autotexts = ax.pie(
            data, 
            labels=None,  # No direct labels, we'll use legend
            autopct=make_autopct(percentages),  # Only show percentages for segments >= min_percent
            colors=colors,
            startangle=90,
            wedgeprops={'width': 0.5, 'edgecolor': 'w'},
            textprops={'fontsize': config["annotation_size"]}
        )
        
        # Customize percentage text appearance inside pi chart
        for autotext in autotexts:
            autotext.set_fontsize(config["annotation_size"])

        
        # Add legend with percentages
        legend_labels = [f"{labels[i]} ({pct:.1f}%)" for i, pct in enumerate(percentages)]
        legend = ax.legend(
            wedges, 
            legend_labels, 
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),  # Slightly more separation
            fontsize=config["legend_size"],
            framealpha=0.7  # Semi-transparent frame
        )
        
        # Add title with proper padding
        ax.set_title(title, fontsize=config["title_size"], pad=15)
        
        # Add donut hole
        centre_circle = plt.Circle((0, 0), 0.25, fc='white') 
        ax.add_patch(centre_circle)
        
        # Ensure the legend fits within the figure
        plt.tight_layout()
        return fig
    
    @classmethod
    def create_bar_chart(cls, data, labels, title, xlabel, ylabel, size=MEDIUM):
        """Create a standardized bar chart"""
        config = cls.SIZE_CONFIG.get(size, cls.SIZE_CONFIG[cls.MEDIUM])
        
        # Create figure
        fig, ax = plt.subplots(figsize=config["figsize"]["bar"], dpi=config["dpi"])
        
        # Plot bars
        bars = ax.bar(labels, data, color='steelblue')
        
        # Add title and labels
        ax.set_title(title, fontsize=config["title_size"])
        ax.set_xlabel(xlabel, fontsize=config["label_size"])
        ax.set_ylabel(ylabel, fontsize=config["label_size"])
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=config["tick_size"])
        plt.yticks(fontsize=config["tick_size"])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.1, 
                str(int(height)), 
                ha='center', 
                fontsize=config["annotation_size"]
            )
        
        plt.tight_layout()
        return fig
    
    @classmethod
    def create_line_chart(cls, x_data, y_data, title, xlabel, ylabel, size=MEDIUM):
        """Create a standardized line chart"""
        config = cls.SIZE_CONFIG.get(size, cls.SIZE_CONFIG[cls.MEDIUM])
        
        # Create figure
        fig, ax = plt.subplots(figsize=config["figsize"]["line"], dpi=config["dpi"])
        
        # Plot line
        ax.plot(x_data, y_data, marker='o', linestyle='-', color='darkblue')
        
        # Add title and labels
        ax.set_title(title, fontsize=config["title_size"])
        ax.set_xlabel(xlabel, fontsize=config["label_size"])
        ax.set_ylabel(ylabel, fontsize=config["label_size"])
        
        # Set tick size
        plt.xticks(fontsize=config["tick_size"])
        plt.yticks(fontsize=config["tick_size"])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig

    @classmethod
    def figure_to_bytes(cls, fig):
        """Convert figure to bytes for downloading"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()