import { forwardRef } from "react";
import { useFighterSearch } from "../hooks/useFighterSearch";
import type { FighterSearchResult } from "../types";

interface FighterCardProps {
  corner: "a" | "b";
  isWinner: boolean;
  onSelect: (fighter: FighterSearchResult | null) => void;
}

export const FighterCard = forwardRef<HTMLInputElement, FighterCardProps>(
  function FighterCard({ corner, isWinner, onSelect }, inputRef) {
    const search = useFighterSearch(onSelect);

    const inputId = `fighter-${corner}`;
    const listId = `suggestions-${corner}`;

    return (
      <div className={`fighter-card corner-${corner}${isWinner ? " winner" : ""}`}>
        <span className={`badge badge-${corner}`}>Corner {corner.toUpperCase()}</span>
        <label htmlFor={inputId}>Fighter</label>
        <div className="search-wrap">
          <input
            id={inputId}
            ref={inputRef}
            type="text"
            placeholder="Search fighters…"
            autoComplete="off"
            spellCheck={false}
            role="combobox"
            aria-expanded={search.isOpen}
            aria-autocomplete="list"
            aria-controls={listId}
            aria-activedescendant={
              search.activeIndex >= 0 ? `suggestion-${corner}-${search.activeIndex}` : undefined
            }
            value={search.query}
            onChange={search.handleChange}
            onKeyDown={search.handleKeyDown}
            onBlur={search.handleBlur}
          />
          <ul
            className="suggestions"
            id={listId}
            role="listbox"
            aria-label={`Corner ${corner.toUpperCase()} fighter suggestions`}
            hidden={!search.isOpen}
          >
            {search.suggestions.map((fighter, i) => (
              <li
                key={fighter.name}
                id={`suggestion-${corner}-${i}`}
                role="option"
                aria-selected={i === search.activeIndex}
                onMouseDown={(e) => {
                  e.preventDefault();
                  search.choose(fighter);
                }}
              >
                <strong>{fighter.name}</strong>
                <span>UFC {fighter.record}</span>
              </li>
            ))}
          </ul>
        </div>
        <p className="picked">
          {search.selected
            ? `UFC record: ${search.selected.record} · ${search.selected.fights} fights`
            : ""}
        </p>
      </div>
    );
  },
);
