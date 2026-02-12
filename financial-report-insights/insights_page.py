"""
Dynamic Financial Insights Dashboard Page.
Interactive Streamlit page with Plotly visualizations for CFO-grade financial insights.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st
import pandas as pd
import numpy as np

from config import settings
from excel_processor import ExcelProcessor, WorkbookData, SheetData
from financial_analyzer import CharlieAnalyzer, FinancialData, quick_analyze
from viz_utils import FinancialVizUtils

logger = logging.getLogger(__name__)


class FinancialInsightsPage:
    """
    Dynamic financial insights dashboard with:
    - Executive summary cards
    - Interactive charts
    - Drill-down capabilities
    - Natural language query integration
    """

    def __init__(self, docs_folder: str = "./documents"):
        """Initialize the insights page."""
        self.docs_folder = Path(docs_folder)
        self.analyzer = CharlieAnalyzer()
        self.viz = FinancialVizUtils()
        self.processor = ExcelProcessor(docs_folder)

    def render(self):
        """Main page render method."""
        st.title("Financial Insights Dashboard")
        st.markdown("*Powered by Charlie-style Analysis*")

        # Load data
        excel_files = self.processor.scan_for_excel_files()

        if not excel_files:
            st.warning("No Excel files found in the documents folder.")
            st.info("Upload Excel files (.xlsx, .xlsm, .csv) to the documents folder to begin analysis.")

            # Show upload option
            self._render_upload_section()
            return

        # Sidebar: Data selection
        with st.sidebar:
            selected_file, selected_sheet = self._render_data_selector(excel_files)
            analysis_options = self._render_analysis_options()

        # Load selected data
        if selected_file:
            try:
                workbook = self.processor.load_workbook(selected_file)
                combined = self.processor.combine_sheets_intelligently(workbook)

                # Get the selected sheet's data
                df = self._get_sheet_data(workbook, selected_sheet)

                if df is not None and not df.empty:
                    # Store in session state for cross-tab access
                    st.session_state['current_df'] = df
                    st.session_state['current_workbook'] = workbook
                    st.session_state['analysis_results'] = self.analyzer.analyze(df)

                    # Main content tabs
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "Executive Summary",
                        "Financial Ratios",
                        "Trends & Forecasts",
                        "Budget Analysis",
                        "Cash Flow",
                        "Scoring Models",
                    ])

                    with tab1:
                        self._render_executive_summary(df, workbook)
                    with tab2:
                        self._render_ratios_dashboard(df)
                    with tab3:
                        self._render_trends_dashboard(df, workbook)
                    with tab4:
                        self._render_budget_dashboard(df, workbook)
                    with tab5:
                        self._render_cashflow_dashboard(df)
                    with tab6:
                        self._render_scoring_models(df)

                    # Report download
                    self._render_report_download(df)

                    # Data explorer at bottom
                    self._render_data_explorer(df, workbook)

                else:
                    st.warning("No data available in the selected sheet.")

            except Exception as e:
                st.error(f"Error loading data: {e}")
                logger.exception("Failed to load Excel data")

    def _render_data_selector(self, excel_files: List[Path]) -> tuple:
        """Render the data source selector in sidebar."""
        st.header("Data Source")

        # File selection
        file_names = [f.name for f in excel_files]
        selected_name = st.selectbox("Select File", file_names)
        selected_file = excel_files[file_names.index(selected_name)] if selected_name else None

        # Sheet selection (reuse cached workbook to avoid double load)
        selected_sheet = None
        if selected_file:
            try:
                cache_key = f"_wb_{selected_file}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = self.processor.load_workbook(selected_file)
                workbook = st.session_state[cache_key]
                sheet_names = [s.name for s in workbook.sheets]
                if sheet_names:
                    selected_sheet = st.selectbox("Select Sheet", ["All Sheets"] + sheet_names)
            except Exception as e:
                st.error(f"Error reading sheets: {e}")

        # File info
        if selected_file:
            st.caption(f"Size: {selected_file.stat().st_size / 1024:.1f} KB")

        return selected_file, selected_sheet

    def _render_analysis_options(self) -> Dict[str, Any]:
        """Render analysis options in sidebar."""
        st.header("Analysis Options")

        options = {
            'show_insights': st.checkbox("Show AI Insights", value=True),
            'show_benchmarks': st.checkbox("Show Benchmarks", value=True),
            'currency': st.selectbox("Currency", ["$", "€", "£", "¥"], index=0),
            'decimal_places': st.slider("Decimal Places", 0, 4, 2)
        }

        # Refresh button
        if st.button("Refresh Analysis"):
            # Clear cached data
            for key in ['current_df', 'current_workbook', 'analysis_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        return options

    def _get_sheet_data(self, workbook: WorkbookData, selected_sheet: Optional[str]) -> pd.DataFrame:
        """Get data for the selected sheet."""
        if not workbook.sheets:
            return pd.DataFrame()

        if selected_sheet == "All Sheets" or selected_sheet is None:
            # Combine all sheets
            dfs = [s.df for s in workbook.sheets]
            if dfs:
                # Add source column
                for i, (df, sheet) in enumerate(zip(dfs, workbook.sheets)):
                    dfs[i] = df.copy()
                    dfs[i]['_source_sheet'] = sheet.name
                return pd.concat(dfs, ignore_index=True)
            return pd.DataFrame()

        # Find specific sheet
        for sheet in workbook.sheets:
            if sheet.name == selected_sheet:
                return sheet.df

        return pd.DataFrame()

    def _render_upload_section(self):
        """Render file upload section."""
        st.subheader("Upload Financial Data")

        uploaded_files = st.file_uploader(
            "Upload Excel or CSV files",
            type=['xlsx', 'xlsm', 'xls', 'csv'],
            accept_multiple_files=True
        )

        if uploaded_files:
            max_size = settings.max_file_size_mb * 1024 * 1024
            for file in uploaded_files:
                safe_name = os.path.basename(file.name).strip()
                if not safe_name or safe_name in (".", ".."):
                    st.error(f"Invalid filename: {file.name}")
                    continue
                save_path = (self.docs_folder / safe_name).resolve()
                if not str(save_path).startswith(str(self.docs_folder.resolve())):
                    st.error(f"Rejected path traversal attempt: {file.name}")
                    continue
                file.seek(0, 2)
                size = file.tell()
                file.seek(0)
                if size > max_size:
                    st.error(f"File too large: {safe_name} ({size / 1024 / 1024:.1f} MB)")
                    continue
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"Uploaded: {safe_name}")

            if st.button("Load Uploaded Files"):
                st.rerun()

    def _render_executive_summary(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render executive summary with KPI cards and key insights."""
        st.subheader("Executive Summary")

        # Extract key metrics from analysis
        analysis = st.session_state.get('analysis_results', {})

        # KPI Cards row
        col1, col2, col3, col4 = st.columns(4)

        # Try to find key financial metrics
        metrics = self._extract_key_metrics(df)

        with col1:
            if metrics.get('revenue'):
                st.metric(
                    label="Revenue",
                    value=self.viz.format_currency(metrics['revenue']),
                    delta=f"{metrics.get('revenue_growth', 0):.1%}" if metrics.get('revenue_growth') else None
                )
            else:
                st.metric(label="Total Records", value=f"{len(df):,}")

        with col2:
            if metrics.get('net_income'):
                st.metric(
                    label="Net Income",
                    value=self.viz.format_currency(metrics['net_income']),
                    delta=f"{metrics.get('net_income_growth', 0):.1%}" if metrics.get('net_income_growth') else None
                )
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.metric(label=f"Avg {numeric_cols[0]}", value=f"{df[numeric_cols[0]].mean():,.2f}")

        with col3:
            prof_ratios = analysis.get('profitability_ratios', {})
            if prof_ratios.get('operating_margin'):
                st.metric(
                    label="Operating Margin",
                    value=f"{prof_ratios['operating_margin']:.1%}"
                )
            elif prof_ratios.get('net_margin'):
                st.metric(
                    label="Net Margin",
                    value=f"{prof_ratios['net_margin']:.1%}"
                )
            else:
                st.metric(label="Columns", value=len(df.columns))

        with col4:
            cf_analysis = analysis.get('cash_flow')
            if cf_analysis and cf_analysis.free_cash_flow:
                st.metric(
                    label="Free Cash Flow",
                    value=self.viz.format_currency(cf_analysis.free_cash_flow)
                )
            else:
                st.metric(label="Sheets", value=len(workbook.sheets))

        st.divider()

        # AI-generated insights
        insights = analysis.get('insights', [])
        if insights:
            st.subheader("AI-Generated Insights")

            for insight in insights[:5]:
                severity_color = {
                    'info': 'blue',
                    'warning': 'orange',
                    'critical': 'red'
                }.get(insight.severity, 'gray')

                with st.container():
                    st.markdown(f"**{insight.category.upper()}**: {insight.message}")
                    if insight.recommendation:
                        st.caption(f"*Recommendation: {insight.recommendation}*")

        # Data overview
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Detected Financial Types:**")
            for sheet in workbook.sheets:
                st.caption(f"• {sheet.name}: {sheet.detected_type or 'Custom data'}")

        with col2:
            st.markdown("**Numeric Columns Available:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
            for col in numeric_cols:
                st.caption(f"• {col}")

    def _render_ratios_dashboard(self, df: pd.DataFrame):
        """Render financial ratios visualization."""
        st.subheader("Financial Ratios Analysis")

        analysis = st.session_state.get('analysis_results', {})

        col1, col2 = st.columns(2)

        with col1:
            # Liquidity ratios
            st.markdown("**Liquidity Ratios**")
            liquidity = analysis.get('liquidity_ratios', {})

            if any(v is not None for v in liquidity.values()):
                fig = self.viz.create_ratio_dashboard(
                    {k: v for k, v in liquidity.items() if v is not None},
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Liquidity ratios require balance sheet data (Current Assets, Current Liabilities, Inventory, Cash)")

        with col2:
            # Profitability ratios
            st.markdown("**Profitability Ratios**")
            profitability = analysis.get('profitability_ratios', {})

            if any(v is not None for v in profitability.values()):
                # Filter out None values and format for display
                profit_display = {k: v for k, v in profitability.items() if v is not None}
                fig = self.viz.create_ratio_dashboard(profit_display, title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Profitability ratios require income statement data (Revenue, COGS, Operating Income, Net Income)")

        # Leverage and Efficiency
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Leverage Ratios**")
            leverage = analysis.get('leverage_ratios', {})

            if any(v is not None for v in leverage.values()):
                lev_display = {k: v for k, v in leverage.items() if v is not None}
                fig = self.viz.create_ratio_dashboard(lev_display, title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Leverage ratios require debt and equity data")

        with col4:
            st.markdown("**Efficiency Ratios**")
            efficiency = analysis.get('efficiency_ratios', {})

            if any(v is not None for v in efficiency.values()):
                eff_display = {k: v for k, v in efficiency.items() if v is not None}
                fig = self.viz.create_ratio_dashboard(eff_display, title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Efficiency ratios require revenue, inventory, and receivables data")

        # Ratio definitions
        with st.expander("Ratio Definitions"):
            for name, defn in self.analyzer.ratio_definitions.items():
                st.markdown(f"**{defn['name']}**: {defn['formula']}")
                st.caption(defn['interpretation'])

    def _render_trends_dashboard(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render trends and forecasts visualization."""
        st.subheader("Trends & Forecasts")

        # Select metric for trend analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns available for trend analysis.")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_metric = st.selectbox("Select Metric", numeric_cols)
            forecast_periods = st.slider("Forecast Periods", 1, 12, 3)
            forecast_method = st.selectbox(
                "Forecast Method",
                ["linear", "moving_average", "growth_rate"]
            )

        # Check if there's a time/period column
        potential_time_cols = [col for col in df.columns if any(
            pattern in col.lower() for pattern in ['date', 'period', 'month', 'year', 'quarter']
        )]

        period_col = None
        if potential_time_cols:
            period_col = st.selectbox("Period Column (optional)", ["None"] + potential_time_cols)
            if period_col == "None":
                period_col = None

        with col2:
            # Create trend visualization
            values = df[selected_metric].dropna().tolist()

            if len(values) >= 2:
                # Trend analysis
                if period_col:
                    periods = df[period_col].astype(str).tolist()
                else:
                    periods = [f"Period {i+1}" for i in range(len(values))]

                # Generate forecast
                forecast = self.analyzer.forecast_simple(values, forecast_periods, forecast_method)

                fig = self.viz.create_trend_chart(
                    periods=periods,
                    values=values,
                    title=f"{selected_metric} Trend",
                    show_forecast=True,
                    forecast_periods=forecast.forecast_periods,
                    forecast_values=forecast.forecasted_values
                )
                st.plotly_chart(fig, use_container_width=True)

                # Trend statistics
                trend = self.analyzer.analyze_trends(
                    pd.DataFrame({selected_metric: values}),
                    selected_metric
                )

                st.markdown("**Trend Statistics:**")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    if trend.cagr is not None:
                        st.metric("CAGR", f"{trend.cagr:.1%}")
                with stats_col2:
                    if trend.yoy_growth is not None:
                        st.metric("YoY Growth", f"{trend.yoy_growth:.1%}")
                with stats_col3:
                    st.metric("Trend Direction", trend.trend_direction.title())
                with stats_col4:
                    st.metric("Seasonality", "Detected" if trend.seasonality_detected else "Not detected")

            else:
                st.warning("Need at least 2 data points for trend analysis.")

        # Multi-metric comparison
        st.subheader("Multi-Metric Comparison")

        selected_metrics = st.multiselect(
            "Select metrics to compare",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )

        if selected_metrics and len(df) > 1:
            # Create comparison chart
            chart_df = df[selected_metrics].copy()
            chart_df['Period'] = range(len(chart_df))

            fig = self.viz.create_time_series(
                chart_df,
                'Period',
                selected_metrics,
                title="Metric Comparison",
                show_markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_budget_dashboard(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render budget vs actual analysis."""
        st.subheader("Budget vs Actual Analysis")

        # Try to detect budget and actual columns
        cols = df.columns.tolist()

        actual_cols = [c for c in cols if 'actual' in c.lower()]
        budget_cols = [c for c in cols if 'budget' in c.lower() or 'plan' in c.lower() or 'target' in c.lower()]
        item_cols = [c for c in cols if any(x in c.lower() for x in ['item', 'category', 'account', 'description', 'name'])]

        if not (actual_cols and budget_cols):
            st.info("""
            Budget analysis requires columns with 'Actual' and 'Budget' (or 'Plan'/'Target') in their names.

            **Tips for budget analysis:**
            - Ensure your data has columns like 'Actual', 'Budget', 'Variance'
            - Include a category/item column for line-item analysis
            - Upload a budget vs actual report for full functionality
            """)

            # Show general variance if numeric columns exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                st.subheader("Manual Variance Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    actual_col = st.selectbox("Select Actual Column", numeric_cols)
                with col2:
                    budget_col = st.selectbox("Select Budget Column", [c for c in numeric_cols if c != actual_col])

                if st.button("Calculate Variance"):
                    variance = df[actual_col].sum() - df[budget_col].sum()
                    variance_pct = variance / df[budget_col].sum() if df[budget_col].sum() != 0 else 0

                    st.metric(
                        "Total Variance",
                        self.viz.format_currency(variance),
                        f"{variance_pct:.1%}"
                    )
            return

        # Auto-detected budget analysis
        col1, col2, col3 = st.columns(3)

        with col1:
            actual_col = st.selectbox("Actual Column", actual_cols)
        with col2:
            budget_col = st.selectbox("Budget Column", budget_cols)
        with col3:
            item_col = st.selectbox("Item Column", item_cols if item_cols else cols[:5])

        # Calculate variances
        df_analysis = df[[item_col, actual_col, budget_col]].dropna()
        df_analysis['Variance'] = df_analysis[actual_col] - df_analysis[budget_col]
        df_analysis['Variance_Pct'] = df_analysis['Variance'] / df_analysis[budget_col].abs() * 100

        # Summary metrics
        total_actual = df_analysis[actual_col].sum()
        total_budget = df_analysis[budget_col].sum()
        total_variance = total_actual - total_budget
        variance_pct = total_variance / total_budget if total_budget != 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Actual", self.viz.format_currency(total_actual))
        with m2:
            st.metric("Total Budget", self.viz.format_currency(total_budget))
        with m3:
            st.metric("Total Variance", self.viz.format_currency(total_variance))
        with m4:
            st.metric("Variance %", f"{variance_pct:.1%}")

        # Waterfall chart
        st.subheader("Variance Waterfall")

        # Get top variances for waterfall
        top_variances = df_analysis.nlargest(10, 'Variance', keep='first')

        fig = self.viz.create_waterfall(
            categories=top_variances[item_col].tolist(),
            values=top_variances['Variance'].tolist(),
            title="Top Variances by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Favorable Variances (Under Budget)**")
            favorable = df_analysis[df_analysis['Variance'] < 0].sort_values('Variance')
            if not favorable.empty:
                st.dataframe(favorable.head(10), use_container_width=True)
            else:
                st.caption("No favorable variances")

        with col2:
            st.markdown("**Unfavorable Variances (Over Budget)**")
            unfavorable = df_analysis[df_analysis['Variance'] > 0].sort_values('Variance', ascending=False)
            if not unfavorable.empty:
                st.dataframe(unfavorable.head(10), use_container_width=True)
            else:
                st.caption("No unfavorable variances")

    def _render_cashflow_dashboard(self, df: pd.DataFrame):
        """Render cash flow and working capital analysis."""
        st.subheader("Cash Flow & Working Capital")

        analysis = st.session_state.get('analysis_results', {})
        cf_analysis = analysis.get('cash_flow')
        wc_analysis = analysis.get('working_capital')

        # Cash Flow Metrics
        st.markdown("**Cash Flow Metrics**")

        if cf_analysis:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if cf_analysis.operating_cf is not None:
                    st.metric("Operating CF", self.viz.format_currency(cf_analysis.operating_cf))
            with col2:
                if cf_analysis.free_cash_flow is not None:
                    st.metric("Free Cash Flow", self.viz.format_currency(cf_analysis.free_cash_flow))
            with col3:
                if cf_analysis.dso is not None:
                    st.metric("DSO", f"{cf_analysis.dso:.0f} days")
            with col4:
                if cf_analysis.cash_conversion_cycle is not None:
                    st.metric("Cash Conversion Cycle", f"{cf_analysis.cash_conversion_cycle:.0f} days")

            # Cash Conversion Cycle gauge
            if cf_analysis.cash_conversion_cycle is not None:
                fig = self.viz.create_gauge_chart(
                    value=cf_analysis.cash_conversion_cycle,
                    title="Cash Conversion Cycle (Days)",
                    min_val=0,
                    max_val=120,
                    thresholds={'warning': 60, 'good': 30}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Working capital cycle components
            if cf_analysis.dso is not None or cf_analysis.dio is not None or cf_analysis.dpo is not None:
                st.markdown("**Working Capital Cycle Components**")
                components = []
                values = []

                if cf_analysis.dso:
                    components.append("DSO\n(Days Sales Outstanding)")
                    values.append(cf_analysis.dso)
                if cf_analysis.dio:
                    components.append("DIO\n(Days Inventory Outstanding)")
                    values.append(cf_analysis.dio)
                if cf_analysis.dpo:
                    components.append("DPO\n(Days Payables Outstanding)")
                    values.append(cf_analysis.dpo)

                if components:
                    fig = self.viz.create_comparison_bar(
                        categories=components,
                        values1=values,
                        values2=[45] * len(values),  # Benchmark
                        name1="Actual",
                        name2="Benchmark",
                        title="Working Capital Cycle Days"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("""
            Cash flow analysis requires specific data:
            - Operating Cash Flow
            - Investing Cash Flow
            - Financing Cash Flow
            - Accounts Receivable
            - Inventory
            - Accounts Payable
            - Revenue and COGS

            Upload a cash flow statement or balance sheet for full functionality.
            """)

        # Working Capital Analysis
        st.markdown("**Working Capital Analysis**")

        if wc_analysis and wc_analysis.net_working_capital is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Current Assets", self.viz.format_currency(wc_analysis.current_assets or 0))
            with col2:
                st.metric("Current Liabilities", self.viz.format_currency(wc_analysis.current_liabilities or 0))
            with col3:
                st.metric("Net Working Capital", self.viz.format_currency(wc_analysis.net_working_capital))

    def _render_scoring_models(self, df: pd.DataFrame):
        """Render Scoring Models dashboard: DuPont, Z-Score, F-Score, Composite Health."""
        st.subheader("Scoring Models")

        analysis = st.session_state.get('analysis_results', {})
        dupont = analysis.get('dupont')
        z_result = analysis.get('altman_z_score')
        f_result = analysis.get('piotroski_f_score')
        health = analysis.get('composite_health')

        # --- Row 1: Composite Health Score ---
        if health is not None:
            st.markdown("**Composite Financial Health**")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.metric("Health Score", f"{health.score}/100")
            with col2:
                grade_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'F': 'red'}
                color = grade_colors.get(health.grade, 'gray')
                st.markdown(f"**Grade:** :{color}[**{health.grade}**]")
            with col3:
                if health.interpretation:
                    st.info(health.interpretation)

            # Component breakdown bar chart
            if health.component_scores:
                import plotly.graph_objects as go
                components = list(health.component_scores.keys())
                values = list(health.component_scores.values())
                max_pts = {'z_score': 25, 'f_score': 25, 'profitability': 20,
                           'liquidity': 15, 'leverage': 15}
                maxes = [max_pts.get(c, 25) for c in components]
                labels = [c.replace('_', ' ').title() for c in components]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=labels, y=values, name='Score',
                    marker_color='steelblue', text=values, textposition='auto',
                ))
                fig.add_trace(go.Bar(
                    x=labels, y=[m - v for m, v in zip(maxes, values)],
                    name='Remaining', marker_color='lightgray',
                ))
                fig.update_layout(
                    barmode='stack', title='Health Score Components',
                    yaxis_title='Points', height=300,
                    showlegend=False, margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

        # --- Row 2: Altman Z-Score and Piotroski F-Score side by side ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Altman Z-Score**")
            if z_result and z_result.z_score is not None:
                zone_colors = {'safe': 'green', 'grey': 'orange', 'distress': 'red', 'partial': 'gray'}
                zone_color = zone_colors.get(z_result.zone, 'gray')
                st.metric("Z-Score", f"{z_result.z_score:.2f}")
                st.markdown(f"Zone: :{zone_color}[**{z_result.zone.title()}**]")

                # Z-Score gauge
                fig = self.viz.create_gauge_chart(
                    value=z_result.z_score,
                    title="Altman Z-Score",
                    min_val=0, max_val=5,
                    thresholds={'warning': 1.81, 'good': 2.99},
                )
                st.plotly_chart(fig, use_container_width=True)

                # Components table
                if z_result.components:
                    st.markdown("**Components:**")
                    comp_labels = {
                        'x1': 'Working Capital / Assets',
                        'x2': 'Retained Earnings / Assets',
                        'x3': 'EBIT / Assets',
                        'x4': 'Equity / Liabilities',
                        'x5': 'Sales / Assets',
                    }
                    for k, v in z_result.components.items():
                        if v is not None:
                            label = comp_labels.get(k, k)
                            st.caption(f"{k.upper()}: {label} = {v:.4f}")
            else:
                st.info("Insufficient data for Z-Score calculation. Requires total assets, "
                        "current assets/liabilities, retained earnings, EBIT, equity, "
                        "liabilities, and revenue.")

        with col_right:
            st.markdown("**Piotroski F-Score**")
            if f_result:
                st.metric("F-Score", f"{f_result.score}/{f_result.max_score}")

                if f_result.score >= 7:
                    strength = "Strong"
                    strength_color = "green"
                elif f_result.score >= 4:
                    strength = "Moderate"
                    strength_color = "orange"
                else:
                    strength = "Weak"
                    strength_color = "red"
                st.markdown(f"Strength: :{strength_color}[**{strength}**]")

                # Criteria checklist
                if f_result.criteria:
                    st.markdown("**Criteria:**")
                    for criterion, passed in f_result.criteria.items():
                        icon = "+" if passed else "-"
                        label = criterion.replace('_', ' ').title()
                        st.caption(f"{icon} {label}")
            else:
                st.info("Insufficient data for F-Score calculation.")

        st.divider()

        # --- Row 3: DuPont Decomposition ---
        st.markdown("**DuPont ROE Decomposition**")
        if dupont and dupont.roe is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("ROE", f"{dupont.roe:.1%}")
            with col2:
                if dupont.net_margin is not None:
                    st.metric("Net Margin", f"{dupont.net_margin:.1%}")
            with col3:
                if dupont.asset_turnover is not None:
                    st.metric("Asset Turnover", f"{dupont.asset_turnover:.2f}x")
            with col4:
                if dupont.equity_multiplier is not None:
                    st.metric("Equity Multiplier", f"{dupont.equity_multiplier:.2f}x")

            if dupont.primary_driver:
                st.caption(f"Primary ROE driver: {dupont.primary_driver.replace('_', ' ').title()}")

            # 5-factor extensions
            if dupont.tax_burden is not None or dupont.interest_burden is not None:
                st.markdown("**5-Factor Extensions:**")
                ext_col1, ext_col2 = st.columns(2)
                with ext_col1:
                    if dupont.tax_burden is not None:
                        st.metric("Tax Burden (NI/EBT)", f"{dupont.tax_burden:.1%}")
                with ext_col2:
                    if dupont.interest_burden is not None:
                        st.metric("Interest Burden (EBT/EBIT)", f"{dupont.interest_burden:.1%}")

            if dupont.interpretation:
                st.info(dupont.interpretation)
        else:
            st.info("DuPont analysis requires net income, revenue, total assets, and total equity.")

        # --- Row 4: Industry Benchmark Comparison ---
        st.divider()
        self._render_industry_benchmarks(analysis)

    # Industry benchmarks (general cross-industry averages)
    INDUSTRY_BENCHMARKS = {
        'current_ratio': {'label': 'Current Ratio', 'benchmark': 1.5, 'good': 2.0, 'unit': 'x'},
        'quick_ratio': {'label': 'Quick Ratio', 'benchmark': 1.0, 'good': 1.5, 'unit': 'x'},
        'net_margin': {'label': 'Net Margin', 'benchmark': 0.08, 'good': 0.15, 'unit': '%'},
        'roe': {'label': 'Return on Equity', 'benchmark': 0.12, 'good': 0.20, 'unit': '%'},
        'roa': {'label': 'Return on Assets', 'benchmark': 0.06, 'good': 0.10, 'unit': '%'},
        'debt_to_equity': {'label': 'Debt to Equity', 'benchmark': 1.0, 'good': 0.5, 'unit': 'x', 'lower_is_better': True},
        'interest_coverage': {'label': 'Interest Coverage', 'benchmark': 3.0, 'good': 6.0, 'unit': 'x'},
        'asset_turnover': {'label': 'Asset Turnover', 'benchmark': 0.8, 'good': 1.2, 'unit': 'x'},
        'gross_margin': {'label': 'Gross Margin', 'benchmark': 0.35, 'good': 0.50, 'unit': '%'},
        'operating_margin': {'label': 'Operating Margin', 'benchmark': 0.10, 'good': 0.20, 'unit': '%'},
    }

    def _render_industry_benchmarks(self, analysis: Dict[str, Any]):
        """Render industry benchmark comparisons for key financial ratios."""
        st.markdown("**Industry Benchmark Comparison**")

        # Collect company ratios from analysis results
        company_ratios = {}
        for category_key in ('liquidity_ratios', 'profitability_ratios', 'leverage_ratios', 'efficiency_ratios'):
            ratios = analysis.get(category_key, {})
            for key, value in ratios.items():
                if value is not None:
                    company_ratios[key] = value

        if not company_ratios:
            st.info("No ratio data available for benchmark comparison.")
            return

        # Build comparison data
        rows = []
        for ratio_key, bench in self.INDUSTRY_BENCHMARKS.items():
            company_val = company_ratios.get(ratio_key)
            if company_val is None:
                continue

            lower_is_better = bench.get('lower_is_better', False)

            if lower_is_better:
                if company_val <= bench['good']:
                    status = "Above Average"
                elif company_val <= bench['benchmark']:
                    status = "Average"
                else:
                    status = "Below Average"
            else:
                if company_val >= bench['good']:
                    status = "Above Average"
                elif company_val >= bench['benchmark']:
                    status = "Average"
                else:
                    status = "Below Average"

            if bench['unit'] == '%':
                fmt_company = f"{company_val:.1%}"
                fmt_bench = f"{bench['benchmark']:.1%}"
            else:
                fmt_company = f"{company_val:.2f}x"
                fmt_bench = f"{bench['benchmark']:.2f}x"

            rows.append({
                'Metric': bench['label'],
                'Company': fmt_company,
                'Industry Avg': fmt_bench,
                'Status': status,
            })

        if not rows:
            st.info("No matching benchmarks for available ratios.")
            return

        bench_df = pd.DataFrame(rows)

        # Color the status column
        def color_status(val):
            if val == "Above Average":
                return "color: green"
            elif val == "Below Average":
                return "color: red"
            return "color: orange"

        styled = bench_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Plotly comparison chart
        available_benchmarks = [r for r in rows if r['Status'] != '']
        if available_benchmarks:
            import plotly.graph_objects as go

            labels = [r['Metric'] for r in available_benchmarks]
            # Re-extract raw values for charting
            company_vals = []
            bench_vals = []
            for r in available_benchmarks:
                ratio_key = next(
                    k for k, b in self.INDUSTRY_BENCHMARKS.items()
                    if b['label'] == r['Metric']
                )
                cv = company_ratios[ratio_key]
                bv = self.INDUSTRY_BENCHMARKS[ratio_key]['benchmark']
                if self.INDUSTRY_BENCHMARKS[ratio_key]['unit'] == '%':
                    cv *= 100
                    bv *= 100
                company_vals.append(round(cv, 2))
                bench_vals.append(round(bv, 2))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels, y=company_vals,
                name='Company', marker_color='steelblue',
            ))
            fig.add_trace(go.Bar(
                x=labels, y=bench_vals,
                name='Industry Avg', marker_color='lightcoral',
            ))
            fig.update_layout(
                barmode='group', title='Company vs Industry Benchmarks',
                yaxis_title='Value', height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_report_download(self, df: pd.DataFrame):
        """Render a downloadable financial analysis report."""
        st.divider()
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Download Financial Report")
        with col2:
            try:
                financial_data = self.analyzer._dataframe_to_financial_data(df)
                report = self.analyzer.generate_report(financial_data)

                # Build full report text
                report_lines = [
                    "=" * 60,
                    "FINANCIAL ANALYSIS REPORT",
                    f"Generated: {report.generated_at}",
                    "=" * 60,
                    "",
                    "EXECUTIVE SUMMARY",
                    "-" * 40,
                    report.executive_summary,
                    "",
                ]

                for section_key, section_title in [
                    ('ratio_analysis', 'RATIO ANALYSIS'),
                    ('scoring_models', 'SCORING MODELS'),
                    ('risk_assessment', 'RISK ASSESSMENT'),
                    ('recommendations', 'RECOMMENDATIONS'),
                    ('period_comparison', 'PERIOD COMPARISON'),
                ]:
                    if section_key in report.sections:
                        report_lines.extend([
                            section_title,
                            "-" * 40,
                            report.sections[section_key],
                            "",
                        ])

                report_lines.append("=" * 60)
                report_text = "\n".join(report_lines)

                st.download_button(
                    "Download Report",
                    report_text,
                    "financial_report.txt",
                    "text/plain",
                    type="primary",
                )
            except Exception as e:
                logger.debug(f"Could not generate report: {e}")
                st.caption("Report unavailable for this data.")

    def _render_data_explorer(self, df: pd.DataFrame, workbook: WorkbookData):
        """Render data exploration section."""
        with st.expander("Data Explorer", expanded=False):
            st.subheader("Raw Data View")

            # Data filters
            col1, col2 = st.columns(2)

            with col1:
                show_rows = st.slider("Rows to display", 5, 100, 20)
            with col2:
                numeric_only = st.checkbox("Numeric columns only")

            display_df = df.select_dtypes(include=[np.number]) if numeric_only else df

            st.dataframe(display_df.head(show_rows), use_container_width=True)

            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null': df.isnull().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)

            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Data as CSV",
                csv,
                "financial_data.csv",
                "text/csv"
            )

    @staticmethod
    @st.cache_data(ttl=3600)
    def _extract_key_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Extract key financial metrics from DataFrame."""
        metrics = {}

        # Column name patterns to look for
        revenue_patterns = ['revenue', 'sales', 'income', 'turnover']
        income_patterns = ['net income', 'profit', 'earnings']

        for col in df.columns:
            col_lower = col.lower()

            # Revenue
            if any(p in col_lower for p in revenue_patterns) and 'net income' not in col_lower:
                if 'revenue' not in metrics:
                    try:
                        metrics['revenue'] = df[col].sum() if len(df) > 1 else df[col].iloc[0]
                    except (TypeError, ValueError, IndexError):
                        pass

            # Net Income
            if any(p in col_lower for p in income_patterns):
                if 'net_income' not in metrics:
                    try:
                        metrics['net_income'] = df[col].sum() if len(df) > 1 else df[col].iloc[0]
                    except (TypeError, ValueError, IndexError):
                        pass

        return metrics


def render_insights_page(docs_folder: str = "./documents"):
    """Convenience function to render the insights page."""
    page = FinancialInsightsPage(docs_folder)
    page.render()
