# GenNyx Development Session Notes

## Session: 2026-01-19

### Overview
Set up GenNyx MNQ futures paper trading system for Heroku deployment with PostgreSQL persistence.

### Tasks Completed

1. **Project Structure Review**
   - Analyzed Python trading system for MNQ micro nasdaq futures
   - Technical indicators: UT Bot, Supertrend, ADX, EMA, Bollinger Bands
   - Live trading engine with paper trading capability

2. **Database Setup (Neon PostgreSQL)**
   - Created `scripts/create_tables.py` with 8 tables:
     - `trading_state` - Current trading state (JSONB)
     - `trades` - Trade history
     - `daily_stats` - Daily performance metrics
     - `candles` - OHLCV candle data
     - `signals` - Trading signals
     - `positions` - Position tracking
     - `system_config` - System configuration (including OAuth token)
     - `performance_metrics` - Performance tracking
   - Tables created in Neon PostgreSQL QA database

3. **Code Fixes for Database Connectivity**
   - Fixed duplicate `sslmode` parameter in `state_manager.py`
   - Added retry logic with exponential backoff for database connections

4. **Heroku Environment Setup (gennyx-qa)**
   - Set environment variables:
     - `DATABASE_URL` - Neon PostgreSQL connection string
     - `SCHWAB_API_KEY` - API key for Schwab
     - `SCHWAB_API_SECRET` - API secret
     - `SCHWAB_SYMBOL` - /MNQ (set via Heroku API due to CLI issues)
     - `SIMPLE_MODE` - true
     - `LOG_LEVEL` - INFO

5. **Token Storage Solution**
   - Stored Schwab OAuth token in database `system_config` table (key: `schwab_token`)
   - Updated `config.py` with `load_token_from_database()` function
   - Token loading priority: env var -> file -> database

6. **Token Refresh Persistence**
   - Added automatic token refresh detection in `data_feed.py`
   - Token changes saved to database via `_save_token_to_database()`
   - Engine calls `check_and_persist_token()` after state saves

7. **Deployment Test Script**
   - Created `scripts/test_heroku_deploy.py` to validate deployment readiness
   - Tests config loading, data feed, state manager, paper trader, signal generator

8. **Security**
   - Updated `.gitignore` to exclude `config/` directory
   - Sensitive files excluded: credentials.json, token files, .env

9. **GitHub Push**
   - Repository: https://github.com/pugalendhi16/gennyx
   - 26 files, 3,556 lines pushed to main branch

### Key Files Modified
- `gennyx/live/state_manager.py` - Database retry logic
- `gennyx/config.py` - Database token loading
- `gennyx/live/data_feed.py` - Token persistence
- `gennyx/live/engine.py` - Token persistence check
- `scripts/create_tables.py` - Database schema
- `scripts/test_heroku_deploy.py` - Deployment test
- `.gitignore` - Security exclusions

### Database Connection
```
postgresql://neondb_owner:npg_Xxz6nJLTpeB3@ep-misty-art-ahzc67o1-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require
```

### Next Steps
- Deploy to Heroku: `git push heroku main`
- Or connect GitHub to Heroku for automatic deploys
- Monitor logs: `heroku logs --tail -a gennyx-qa`

### Issues Encountered
- Windows Heroku CLI issues with special characters (`/MNQ`, JSON)
- Resolved by using Heroku API via curl for SCHWAB_SYMBOL
- Token storage moved to database as alternative to env var
