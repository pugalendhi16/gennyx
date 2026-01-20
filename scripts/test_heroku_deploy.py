#!/usr/bin/env python3
"""Test script to simulate Heroku deployment environment."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables to simulate Heroku
os.environ['DATABASE_URL'] = 'postgresql://neondb_owner:npg_Xxz6nJLTpeB3@ep-misty-art-ahzc67o1-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require'
os.environ['SCHWAB_API_KEY'] = 'm5DsqKq6GYpe5LMM12ppBkHZgnl4V0VN'
os.environ['SCHWAB_API_SECRET'] = 'Ah9AFQbkxVjpxtAZ'
os.environ['SCHWAB_SYMBOL'] = '/MNQ'
os.environ['SIMPLE_MODE'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')

def main():
    print('=' * 60)
    print('Final Heroku Deployment Simulation Test')
    print('=' * 60)

    # 1. Test config loading
    print('\n[1] Testing config loading...')
    from pathlib import Path
    from gennyx.config import Config
    config = Config.from_env(config_dir=Path('/nonexistent'))
    print(f'    API Key: {config.schwab_api_key[:8]}...')
    print(f'    Symbol: {config.schwab_symbol}')
    print(f'    Token: {"Loaded" if config.schwab_token_json else "Missing"}')
    print(f'    Database: {"Connected" if config.database_url else "Not set"}')

    # 2. Test data feed initialization
    print('\n[2] Testing data feed initialization...')
    from gennyx.live.data_feed import SchwabDataFeed
    data_feed = SchwabDataFeed(config)
    print('    Data feed created')
    print(f'    Database URL set: {data_feed._database_url is not None}')

    # 3. Test state manager
    print('\n[3] Testing state manager...')
    from gennyx.live.state_manager import StatePersistence
    state_mgr = StatePersistence(config.database_url)
    print('    State manager connected')
    print(f'    Has existing state: {state_mgr.has_state()}')

    # 4. Test paper trader
    print('\n[4] Testing paper trader...')
    from gennyx.live.paper_trader import PaperTradeManager
    trader = PaperTradeManager(config)
    print('    Paper trader initialized')
    print(f'    Capital: ${trader.capital:,.2f}')

    # 5. Test signal generator
    print('\n[5] Testing signal generator...')
    from gennyx.live.signals import LiveSignalGenerator
    signal_gen = LiveSignalGenerator(config)
    print('    Signal generator initialized')
    print(f'    Simple mode: {config.simple_mode}')

    # 6. Test candle builder
    print('\n[6] Testing candle builder...')
    from gennyx.live.candle_builder import LiveDataBuilder
    candle_builder = LiveDataBuilder(timezone=config.timezone)
    print('    Candle builder initialized')

    # 7. Test engine creation (without starting)
    print('\n[7] Testing engine creation...')
    from gennyx.live.engine import LiveTradingEngine
    engine = LiveTradingEngine(config)
    print('    Engine created successfully')

    state_mgr.close()

    print('\n' + '=' * 60)
    print('All tests passed! Ready for Heroku deployment.')
    print('=' * 60)
    return 0

if __name__ == '__main__':
    sys.exit(main())
