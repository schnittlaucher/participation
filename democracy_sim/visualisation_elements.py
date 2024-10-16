import matplotlib.pyplot as plt
from mesa.visualization import TextElement
from model_setup import _COLORS
import base64
import math
import io

_COLORS[0] = "LightGray"

def save_plot_to_base64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f'<img src="data:image/png;base64,{image_base64}" />'


class ColorDistributionElement(TextElement):
    def render(self, model):
        # Only render if show_area_stats is enabled
        step = model.scheduler.steps
        if not model.show_area_stats or step == 0 or step % 10 != 0:
            return ""

        # Fetch data from the datacollector
        data = model.datacollector.get_agent_vars_dataframe()
        color_distribution = data['ColorDistribution'].dropna()

        # Extract unique area IDs
        area_ids = color_distribution.index.get_level_values(1).unique()
        num_colors = len(color_distribution.iloc[0])

        # Create subplots within a single figure
        num_areas = len(area_ids)
        num_cols = math.ceil(math.sqrt(num_areas))
        num_rows = math.ceil(num_areas / num_cols)
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                                 figsize=(20, 20), sharex=True)

        for ax, area_id in zip(axes.flatten(), area_ids):
            area_data = color_distribution.xs(area_id, level=1)
            for color_idx in range(num_colors):
                color_data = area_data.apply(lambda x: x[color_idx])
                ax.plot(color_data.index, color_data.values,
                        label=f'Color {color_idx}', color=_COLORS[color_idx])
            ax.set_title(f'Area {area_id}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Color Distribution')
            #ax.legend()

        plt.tight_layout()
        return save_plot_to_base64(fig)


# class VoterTurnoutElement(TextElement):
#     def render(self, model):
#         # Only render if show_area_stats is enabled
#         step = model.scheduler.steps
#         if not model.show_area_stats or step == 0 or step % 10 != 0:
#             return ""
#         # Fetch data from the datacollector
#         data = model.datacollector.get_agent_vars_dataframe()
#         voter_turnout = data['VoterTurnout'].dropna()
#
#         # Extract unique area IDs
#         area_ids = voter_turnout.index.get_level_values(1).unique()
#
#         # Create subplots within a single figure
#         num_areas = len(area_ids)
#         fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), sharex=True)
#
#         for ax, area_id in zip(axes.flatten(), area_ids):
#             area_data = voter_turnout.xs(area_id, level=1)
#             ax.plot(area_data.index, area_data.values, label=f'Area {area_id}')
#             ax.set_title(f'Area {area_id}')
#             ax.set_xlabel('Step')
#             ax.set_ylabel('Voter Turnout (%)')
#             ax.legend()
#
#         plt.tight_layout()
#         return save_plot_to_base64(fig)

class VoterTurnoutElement(TextElement):
    def render(self, model):
        # Only render if show_area_stats is enabled
        step = model.scheduler.steps
        if not model.show_area_stats or step == 0 or step % 10 != 0:
            return ""
        # Fetch data from the datacollector
        data = model.datacollector.get_agent_vars_dataframe()
        voter_turnout = data['VoterTurnout'].dropna()

        # Extract unique area IDs
        area_ids = voter_turnout.index.get_level_values(1).unique()

        # Create a single plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for area_id in area_ids:
            area_data = voter_turnout.xs(area_id, level=1)
            ax.plot(area_data.index, area_data.values, label=f'Area {area_id}')

        ax.set_title('Voter Turnout by Area Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Voter Turnout (%)')
        ax.legend()

        return save_plot_to_base64(fig)


class MatplotlibElement(TextElement):
    def render(self, model):
        # Only render if show_area_stats is enabled
        step = model.scheduler.steps
        if not model.show_area_stats or step == 0 or step % 10 != 0:
            return ""
        # Fetch data from the datacollector
        data = model.datacollector.get_model_vars_dataframe()
        collective_assets = data["Collective assets"]

        # Create a plot
        fig, ax = plt.subplots()
        ax.plot(collective_assets, label="Collective assets")
        ax.set_title("Collective Assets Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Collective Assets")
        ax.legend()

        return save_plot_to_base64(fig)

class StepsTextElement(TextElement):
    def render(self, model):
        step = model.scheduler.steps
        # TODO clean up
        first_agents = [str(a) for a in model.voting_agents[:5]]
        text = (f"Step: {step} | cells: {len(model.color_cells)} | "
                f"areas: {len(model.areas)} | First 5 voters of "
                f"{len(model.voting_agents)}: {first_agents}")
        return text