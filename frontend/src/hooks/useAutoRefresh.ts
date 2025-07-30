import { useEffect, useRef, useCallback } from 'react';

interface UseAutoRefreshOptions {
  interval: number; // in milliseconds
  enabled: boolean;
  onRefresh: () => void;
}

export function useAutoRefresh({ interval, enabled, onRefresh }: UseAutoRefreshOptions) {
  const intervalRef = useRef<number | null>(null);
  const onRefreshRef = useRef(onRefresh);

  // Keep the callback reference up to date
  useEffect(() => {
    onRefreshRef.current = onRefresh;
  }, [onRefresh]);

  const start = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    intervalRef.current = setInterval(() => {
      onRefreshRef.current();
    }, interval);
  }, [interval]);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const restart = useCallback(() => {
    stop();
    if (enabled) {
      start();
    }
  }, [enabled, start, stop]);

  // Start/stop based on enabled state
  useEffect(() => {
    if (enabled) {
      start();
    } else {
      stop();
    }

    return stop;
  }, [enabled, start, stop]);

  // Cleanup on unmount
  useEffect(() => {
    return stop;
  }, [stop]);

  return { start, stop, restart };
}