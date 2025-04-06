import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import os

class TradingLogAnalyzer:
    def __init__(self, log_file_path):
        """Initialize with path to log file"""
        self.log_file_path = log_file_path
        self.sandbox_logs = []
        self.lambda_logs = []
        self.activities_log = None
        self.trades_log = []
        self.trader_data_history = []
        self.products = set()
        
    def parse_logs(self):
        """Parse the different sections of the log file"""
        with open(self.log_file_path, 'r') as f:
            content = f.read()
            
        # Split the content into the three main sections
        sections = content.split("{")
        
        # Extract and parse sandbox logs and lambda logs
        sandbox_lambda_pattern = re.compile(r'{\s*"sandboxLog":\s*"(.*)"\s*,\s*"lambdaLog":\s*"(.*)"\s*,\s*"timestamp":\s*(\d+)\s*}')
        for match in sandbox_lambda_pattern.finditer(content):
            sandbox_log = match.group(1).replace('\\n', '\n')
            lambda_log = match.group(2).replace('\\n', '\n')
            timestamp = int(match.group(3))
            
            self.sandbox_logs.append({
                'timestamp': timestamp,
                'log': sandbox_log
            })
            
            self.lambda_logs.append({
                'timestamp': timestamp,
                'log': lambda_log
            })
            
            # Extract trader data from lambda logs
            trader_data_match = re.search(r'traderData: (.*?)\\n', lambda_log)
            if trader_data_match:
                trader_data = trader_data_match.group(1)
                self.trader_data_history.append({
                    'timestamp': timestamp,
                    'data': trader_data
                })
        
        # Extract activities log
        activities_pattern = re.compile(r'Activities log:(.*?)(?={|\Z)', re.DOTALL)
        activities_match = activities_pattern.search(content)
        if activities_match:
            activities_raw = activities_match.group(1).strip()
            activities_lines = activities_raw.strip().split('\n')
            
            # Parse header and data rows
            if len(activities_lines) > 1:
                header = activities_lines[0].strip().split(';')
                data_rows = []
                
                for line in activities_lines[1:]:
                    if line.strip():  # Skip empty lines
                        values = line.strip().split(';')
                        # Convert values to appropriate types
                        row_dict = {}
                        for i, col in enumerate(header):
                            if i < len(values):
                                # Try to convert to numeric if possible
                                try:
                                    if '.' in values[i]:
                                        row_dict[col] = float(values[i])
                                    else:
                                        row_dict[col] = int(values[i])
                                except ValueError:
                                    row_dict[col] = values[i]
                            else:
                                row_dict[col] = None
                        data_rows.append(row_dict)
                
                self.activities_log = pd.DataFrame(data_rows)
                
                # Extract product names
                if 'product' in self.activities_log.columns:
                    self.products = set(self.activities_log['product'].unique())
        
        # Extract trades
        trades_pattern = re.compile(r'{\s*"timestamp":\s*(\d+),\s*"buyer":\s*"(.*?)",\s*"seller":\s*"(.*?)",\s*"symbol":\s*"(.*?)",\s*"currency":\s*"(.*?)",\s*"price":\s*(\d+),\s*"quantity":\s*(\d+)\s*}')
        for match in trades_pattern.finditer(content):
            self.trades_log.append({
                'timestamp': int(match.group(1)),
                'buyer': match.group(2),
                'seller': match.group(3),
                'symbol': match.group(4),
                'currency': match.group(5),
                'price': int(match.group(6)),
                'quantity': int(match.group(7))
            })
    
    def analyze_market_data(self):
        """Analyze market data from activities log"""
        if self.activities_log is None or self.activities_log.empty:
            print("No activities log data available")
            return None
        
        # Convert to numeric where appropriate
        numeric_columns = [
            'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 
            'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1', 
            'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3',
            'mid_price', 'profit_and_loss'
        ]
        
        for col in numeric_columns:
            if col in self.activities_log.columns:
                self.activities_log[col] = pd.to_numeric(self.activities_log[col], errors='coerce')
        
        # Calculate spread
        if 'ask_price_1' in self.activities_log.columns and 'bid_price_1' in self.activities_log.columns:
            self.activities_log['spread'] = self.activities_log['ask_price_1'] - self.activities_log['bid_price_1']
        
        # Calculate total bid/ask volume
        volume_cols = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
        bid_vols = [col for col in volume_cols if col in self.activities_log.columns]
        if bid_vols:
            self.activities_log['total_bid_volume'] = self.activities_log[bid_vols].sum(axis=1)
        
        volume_cols = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']
        ask_vols = [col for col in volume_cols if col in self.activities_log.columns]
        if ask_vols:
            self.activities_log['total_ask_volume'] = self.activities_log[ask_vols].sum(axis=1)
        
        return self.activities_log
    
    def analyze_trades(self):
        """Analyze trades from trades log"""
        if not self.trades_log:
            print("No trades data available")
            return None
        
        trades_df = pd.DataFrame(self.trades_log)
        
        # Add a flag for your own trades
        trades_df['is_own_trade'] = (trades_df['buyer'] == 'SUBMISSION') | (trades_df['seller'] == 'SUBMISSION')
        
        # For own trades, calculate profit/loss (simplified)
        own_trades = trades_df[trades_df['is_own_trade']].copy()
        own_trades['trade_value'] = own_trades['price'] * own_trades['quantity']
        own_trades['trade_direction'] = np.where(own_trades['buyer'] == 'SUBMISSION', 'BUY', 'SELL')
        own_trades['pnl_impact'] = np.where(own_trades['trade_direction'] == 'BUY', 
                                          -own_trades['trade_value'], 
                                          own_trades['trade_value'])
        
        return {
            'all_trades': trades_df,
            'own_trades': own_trades
        }
    
    def visualize_price_history(self, output_dir='./trading_visualizations'):
        """Visualize price history for each product"""
        if self.activities_log is None or self.activities_log.empty:
            print("No activities log data available for visualization")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # For each product, plot price history
        for product in self.products:
            product_data = self.activities_log[self.activities_log['product'] == product].sort_values('timestamp')
            
            if product_data.empty:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Plot mid price
            if 'mid_price' in product_data.columns:
                plt.plot(product_data['timestamp'], product_data['mid_price'], 
                        label='Mid Price', linewidth=2)
            
            # Plot best bid and ask
            if 'bid_price_1' in product_data.columns:
                plt.plot(product_data['timestamp'], product_data['bid_price_1'], 
                        label='Best Bid', linestyle='--', alpha=0.7)
            
            if 'ask_price_1' in product_data.columns:
                plt.plot(product_data['timestamp'], product_data['ask_price_1'], 
                        label='Best Ask', linestyle='--', alpha=0.7)
            
            # Plot your trades on the same chart
            trades_analysis = self.analyze_trades()
            if trades_analysis and 'own_trades' in trades_analysis:
                own_trades = trades_analysis['own_trades']
                product_trades = own_trades[own_trades['symbol'] == product]
                
                buys = product_trades[product_trades['trade_direction'] == 'BUY']
                sells = product_trades[product_trades['trade_direction'] == 'SELL']
                
                if not buys.empty:
                    plt.scatter(buys['timestamp'], buys['price'], 
                               color='green', marker='^', s=100, label='Your Buys')
                
                if not sells.empty:
                    plt.scatter(sells['timestamp'], sells['price'], 
                               color='red', marker='v', s=100, label='Your Sells')
            
            plt.title(f'{product} Price History')
            plt.xlabel('Timestamp')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{product}_price_history.png')
            plt.close()
    
    def visualize_order_book_depth(self, output_dir='./trading_visualizations'):
        """Visualize order book depth over time"""
        if self.activities_log is None or self.activities_log.empty:
            print("No activities log data available for visualization")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # For each product, plot order book depth
        for product in self.products:
            product_data = self.activities_log[self.activities_log['product'] == product].sort_values('timestamp')
            
            if product_data.empty:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Plot total bid volume as positive
            if 'total_bid_volume' in product_data.columns:
                plt.bar(product_data['timestamp'], product_data['total_bid_volume'], 
                       width=20, alpha=0.6, color='green', label='Bid Volume')
            
            # Plot total ask volume as negative
            if 'total_ask_volume' in product_data.columns:
                plt.bar(product_data['timestamp'], -product_data['total_ask_volume'], 
                       width=20, alpha=0.6, color='red', label='Ask Volume')
            
            plt.title(f'{product} Order Book Depth')
            plt.xlabel('Timestamp')
            plt.ylabel('Volume (Bids positive, Asks negative)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{product}_order_book_depth.png')
            plt.close()
    
    def visualize_trader_data(self, output_dir='./trading_visualizations'):
        """Visualize trader data over time"""
        if not self.trader_data_history:
            print("No trader data available for visualization")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse trader data history
        data_points = []
        for entry in self.trader_data_history:
            timestamp = entry['timestamp']
            data_str = entry['data']
            
            if '|' in data_str:
                parts = data_str.split('|')
                try:
                    resin_fair_value = float(parts[0])
                    
                    # Try to extract kelp prices if available
                    kelp_prices = []
                    if len(parts) > 1 and parts[1]:
                        kelp_prices = [float(p) for p in parts[1].split(',') if p]
                    
                    data_points.append({
                        'timestamp': timestamp,
                        'resin_fair_value': resin_fair_value,
                        'kelp_prices_count': len(kelp_prices),
                        'kelp_latest_price': kelp_prices[-1] if kelp_prices else None
                    })
                except (ValueError, IndexError):
                    pass
        
        if not data_points:
            print("Could not parse any trader data")
            return
            
        df = pd.DataFrame(data_points)
        
        # Plot resin fair value
        plt.figure(figsize=(12, 8))
        plt.plot(df['timestamp'], df['resin_fair_value'], marker='o', linestyle='-', linewidth=2)
        plt.title('Rainforest Resin Fair Value Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Fair Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/resin_fair_value.png')
        plt.close()
        
        # Plot kelp latest price if available
        if 'kelp_latest_price' in df.columns and df['kelp_latest_price'].notna().any():
            plt.figure(figsize=(12, 8))
            plt.plot(df['timestamp'], df['kelp_latest_price'], marker='o', linestyle='-', linewidth=2)
            plt.title('Kelp Latest Price Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/kelp_latest_price.png')
            plt.close()
    
    def generate_performance_report(self, output_dir='./trading_visualizations'):
        """Generate a performance report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze trades
        trades_analysis = self.analyze_trades()
        if not trades_analysis or 'own_trades' not in trades_analysis or trades_analysis['own_trades'].empty:
            print("No trade data available for performance report")
            return
        
        own_trades = trades_analysis['own_trades']
        
        # Calculate overall statistics
        total_trades = len(own_trades)
        total_buy_trades = len(own_trades[own_trades['trade_direction'] == 'BUY'])
        total_sell_trades = len(own_trades[own_trades['trade_direction'] == 'SELL'])
        
        total_buy_volume = own_trades[own_trades['trade_direction'] == 'BUY']['quantity'].sum()
        total_sell_volume = own_trades[own_trades['trade_direction'] == 'SELL']['quantity'].sum()
        
        total_buy_value = own_trades[own_trades['trade_direction'] == 'BUY']['trade_value'].sum()
        total_sell_value = own_trades[own_trades['trade_direction'] == 'SELL']['trade_value'].sum()
        
        # Calculate P&L (very simplified)
        estimated_pnl = total_sell_value - total_buy_value
        
        # Generate report by product
        product_stats = {}
        for product in own_trades['symbol'].unique():
            product_trades = own_trades[own_trades['symbol'] == product]
            
            product_stats[product] = {
                'total_trades': len(product_trades),
                'buy_trades': len(product_trades[product_trades['trade_direction'] == 'BUY']),
                'sell_trades': len(product_trades[product_trades['trade_direction'] == 'SELL']),
                'buy_volume': product_trades[product_trades['trade_direction'] == 'BUY']['quantity'].sum(),
                'sell_volume': product_trades[product_trades['trade_direction'] == 'SELL']['quantity'].sum(),
                'buy_value': product_trades[product_trades['trade_direction'] == 'BUY']['trade_value'].sum(),
                'sell_value': product_trades[product_trades['trade_direction'] == 'SELL']['trade_value'].sum(),
                'estimated_pnl': (product_trades[product_trades['trade_direction'] == 'SELL']['trade_value'].sum() - 
                                 product_trades[product_trades['trade_direction'] == 'BUY']['trade_value'].sum())
            }
        
        # Write report to file
        with open(f'{output_dir}/performance_report.txt', 'w') as f:
            f.write("=== TRADING PERFORMANCE REPORT ===\n\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== OVERALL STATISTICS ===\n")
            f.write(f"Total trades: {total_trades}\n")
            f.write(f"Buy trades: {total_buy_trades}\n")
            f.write(f"Sell trades: {total_sell_trades}\n")
            f.write(f"Total buy volume: {total_buy_volume}\n")
            f.write(f"Total sell volume: {total_sell_volume}\n")
            f.write(f"Total buy value: {total_buy_value}\n")
            f.write(f"Total sell value: {total_sell_value}\n")
            f.write(f"Estimated P&L: {estimated_pnl}\n\n")
            
            f.write("=== PRODUCT STATISTICS ===\n")
            for product, stats in product_stats.items():
                f.write(f"\n--- {product} ---\n")
                f.write(f"Total trades: {stats['total_trades']}\n")
                f.write(f"Buy trades: {stats['buy_trades']}\n")
                f.write(f"Sell trades: {stats['sell_trades']}\n")
                f.write(f"Buy volume: {stats['buy_volume']}\n")
                f.write(f"Sell volume: {stats['sell_volume']}\n")
                f.write(f"Buy value: {stats['buy_value']}\n")
                f.write(f"Sell value: {stats['sell_value']}\n")
                f.write(f"Estimated P&L: {stats['estimated_pnl']}\n")
    
    def run_all_analyses(self, output_dir='./trading_visualizations'):
        """Run all analyses and generate all visualizations"""
        print("Parsing logs...")
        self.parse_logs()
        
        print("Analyzing market data...")
        self.analyze_market_data()
        
        print("Analyzing trades...")
        self.analyze_trades()
        
        print("Generating visualizations...")
        self.visualize_price_history(output_dir)
        self.visualize_order_book_depth(output_dir)
        self.visualize_trader_data(output_dir)
        
        print("Generating performance report...")
        self.generate_performance_report(output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Prosperity trading logs')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output', type=str, default='./trading_visualizations', 
                        help='Path to output directory (default: ./trading_visualizations)')
    
    args = parser.parse_args()
    
    analyzer = TradingLogAnalyzer(args.log_file)
    analyzer.run_all_analyses(args.output)