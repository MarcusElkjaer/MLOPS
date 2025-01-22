import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Search } from 'lucide-react';
import axios from 'axios';

interface SentimentData {
  date: string;
  average_sentiment: number;
}

interface StockData {
  date: string;
  close: number;
}

interface CombinedData {
  date: string;
  average_sentiment?: number;
  close?: number;
}

const apiKey = import.meta.env.VITE_ALPHAVANTAGE_API_KEY || 'your-default-api-key';
const api_url = window.location.origin;

function App() {
  const [searchTerm, setSearchTerm] = useState('');
  const [combinedData, setCombinedData] = useState<CombinedData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const getStockPriceDomain = () => {
    if (!combinedData.length) return [0, 100];
    const prices = combinedData.map(d => d.close || 0).filter(price => price > 0);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    return [
      minPrice - (minPrice * 0.1), // 10% below minimum
      maxPrice + (maxPrice * 0.1)  // 10% above maximum
    ];
  };

  const formatYAxis = (value: number, axis: 'sentiment' | 'price') => {
    if (axis === 'sentiment') {
      return value.toFixed(2);
    } else {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }).format(value);
    }
  };

  const formatTooltipValue = (value: number, name: string) => {
    if (name === 'Sentiment') {
      return value.toFixed(3);
    }
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const fetchData = async () => {
    if (!searchTerm) return;
    
    setLoading(true);
    setError('');
    
    try {
      // Fetch sentiment data
      const sentimentResponse = await axios.get(`${api_url}/get_average_sentiment?search_term=${searchTerm}`);
      const sentimentData: SentimentData[] = sentimentResponse.data;

      // Get date range from sentiment data
      const dates = sentimentData.map(d => new Date(d.date).getTime());
      const minDate = Math.min(...dates);
      const maxDate = Math.max(...dates);

      // Fetch stock data from Alpha Vantage using environment variable
      const stockResponse = await axios.get(
        `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${searchTerm}&apikey=${apiKey}`
      );

      const timeSeriesData = stockResponse.data['Time Series (Daily)'];
      const stockPrices = Object.entries(timeSeriesData)
        .map(([date, values]: [string, any]) => ({
          date,
          close: parseFloat(values['4. close'])
        }))
        .filter(item => {
          const itemDate = new Date(item.date).getTime();
          return itemDate >= minDate && itemDate <= maxDate;
        })
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

      // Create a map of sentiment data for easy lookup
      const sentimentMap = new Map(
        sentimentData.map(item => [item.date, item.average_sentiment])
      );

      // Combine the data using stock prices as the base
      const combined = stockPrices.map(stock => ({
        date: stock.date,
        close: stock.close,
        average_sentiment: sentimentMap.get(stock.date)
      }));

      setCombinedData(combined);
    } catch (err) {
      setError('Failed to fetch data. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchData();
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Stock Sentiment Analysis</h1>
        
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-4">
            <div className="relative flex-1">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Enter stock symbol (e.g., AAPL)"
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <Search className="absolute right-3 top-2.5 text-gray-400" size={20} />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-400"
            >
              {loading ? 'Loading...' : 'Analyze'}
            </button>
          </div>
        </form>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-8">
            {error}
          </div>
        )}

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Stock Price vs Sentiment Analysis</h2>
          <div className="h-[600px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date"
                  angle={-45}
                  textAnchor="end"
                  height={60}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  yAxisId="sentiment"
                  domain={[-1, 1]}
                  tickFormatter={(value) => formatYAxis(value, 'sentiment')}
                  label={{ 
                    value: 'Sentiment Score', 
                    angle: -90, 
                    position: 'insideLeft',
                    style: { textAnchor: 'middle' }
                  }}
                />
                <YAxis 
                  yAxisId="price"
                  orientation="right"
                  domain={getStockPriceDomain()}
                  tickFormatter={(value) => formatYAxis(value, 'price')}
                  label={{ 
                    value: 'Stock Price (USD)', 
                    angle: 90, 
                    position: 'insideRight',
                    style: { textAnchor: 'middle' }
                  }}
                />
                <Tooltip 
                  formatter={formatTooltipValue}
                  labelFormatter={(label) => new Date(label).toLocaleDateString()}
                />
                <Legend verticalAlign="top" height={36} />
                <Line
                  yAxisId="sentiment"
                  type="monotone"
                  dataKey="average_sentiment"
                  stroke="#8884d8"
                  name="Sentiment"
                  dot={false}
                  strokeWidth={2}
                  connectNulls={true}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="close"
                  stroke="#82ca9d"
                  name="Stock Price"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;