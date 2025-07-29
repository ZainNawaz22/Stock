#!/usr/bin/env python3
"""
Start the optimized PSX AI Advisor API server

This script starts the FastAPI server with performance optimizations enabled.
"""

import sys
import os
import argparse
from api_server import run_server

def main():
    """Main function to start the optimized API server."""
    parser = argparse.ArgumentParser(description="Start PSX AI Advisor API Server (Optimized)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print("🚀 Starting PSX AI Advisor API Server (Performance Optimized)")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print("=" * 60)
    print("\n📊 Performance Features Enabled:")
    print("✅ Parallel processing with ThreadPoolExecutor")
    print("✅ Multi-layer caching system")
    print("✅ Request limiting and timeouts")
    print("✅ Background task processing")
    print("✅ Memory optimization and cleanup")
    print("\n🔗 API Endpoints:")
    print(f"   📖 Documentation: http://{args.host}:{args.port}/docs")
    print(f"   🏥 Health Check: http://{args.host}:{args.port}/health")
    print(f"   📈 Stocks: http://{args.host}:{args.port}/api/stocks")
    print(f"   🤖 Predictions: http://{args.host}:{args.port}/api/predictions")
    print(f"   ⚙️  System Status: http://{args.host}:{args.port}/api/system/status")
    print("\n🧪 Test Performance:")
    print(f"   python test_api_performance.py --url http://{args.host}:{args.port}")
    print("\n" + "=" * 60)
    
    try:
        # Start the server
        run_server(host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()