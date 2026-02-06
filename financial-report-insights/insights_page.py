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
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Executive Summary",
                        "Financial Ratios",
                        "Trends & Forecasts",
                        "Budget Analysis",
                        "Cash Flow"
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

        # Sheet selection
        selected_sheet = None
        if selected_file:
            try:
                workbook = self.processor.load_workbook(selected_file)
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
    @st.cache_data(ttl=300)
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
                    except:
                        pass

            # Net Income
            if any(p in col_lower for p in income_patterns):
                if 'net_income' not in metrics:
                    try:
                        metrics['net_income'] = df[col].sum() if len(df) > 1 else df[col].iloc[0]
                    except:
                        pass

        return metrics


def render_insights_page(docs_folder: str = "./documents"):
    """Convenience function to render the insights page."""
    page = FinancialInsightsPage(docs_folder)
    page.render()
