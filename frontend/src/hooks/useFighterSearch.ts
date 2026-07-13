import { useCallback, useEffect, useRef, useState } from "react";
import { searchFighters } from "../api";
import type { FighterSearchResult } from "../types";

/** Ports the setupSearch() closure from the pre-React static/app.js: debounced
 * fetch, keyboard nav, and the aria-activedescendant bookkeeping an accessible
 * combobox needs. */
export function useFighterSearch(onSelect: (fighter: FighterSearchResult | null) => void) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<FighterSearchResult[]>([]);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [selected, setSelected] = useState<FighterSearchResult | null>(null);
  const debounceRef = useRef<number | undefined>(undefined);
  const blurTimeoutRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    return () => {
      window.clearTimeout(debounceRef.current);
      window.clearTimeout(blurTimeoutRef.current);
    };
  }, []);

  const runSearch = useCallback((q: string) => {
    window.clearTimeout(debounceRef.current);
    if (q.length < 2) {
      setSuggestions([]);
      setActiveIndex(-1);
      return;
    }
    debounceRef.current = window.setTimeout(async () => {
      try {
        const results = await searchFighters(q);
        setSuggestions(results);
        setActiveIndex(-1);
      } catch {
        setSuggestions([]);
        setActiveIndex(-1);
      }
    }, 200);
  }, []);

  function choose(fighter: FighterSearchResult) {
    setQuery(fighter.name);
    setSuggestions([]);
    setActiveIndex(-1);
    setSelected(fighter);
    onSelect(fighter);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setQuery(e.target.value);
    setSelected(null);
    onSelect(null);
    runSearch(e.target.value.trim());
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (suggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => (i + 1) % suggestions.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => (i - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === "Enter") {
      if (activeIndex >= 0) {
        e.preventDefault();
        choose(suggestions[activeIndex]);
      }
    } else if (e.key === "Escape") {
      setSuggestions([]);
      setActiveIndex(-1);
    }
  }

  function handleBlur() {
    blurTimeoutRef.current = window.setTimeout(() => {
      setSuggestions([]);
      setActiveIndex(-1);
    }, 150);
  }

  return {
    query,
    suggestions,
    activeIndex,
    selected,
    isOpen: suggestions.length > 0,
    choose,
    handleChange,
    handleKeyDown,
    handleBlur,
  };
}
