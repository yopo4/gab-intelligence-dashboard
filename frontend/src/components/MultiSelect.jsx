import { useState, useRef, useEffect } from "react";

export default function MultiSelect({
  options,
  value,
  onChange,
  placeholder = "Sélectionner…",
  allLabel = "Tous",
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  // Close on outside click
  useEffect(() => {
    function handle(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, []);

  const allSelected = value.length === options.length;
  const label = allSelected
    ? allLabel
    : value.length === 1
      ? value[0]
      : `${value.length} sélectionnés`;

  function toggle(opt) {
    const next = value.includes(opt)
      ? value.filter((v) => v !== opt)
      : [...value, opt];
    onChange(next.length ? next : [...options]);
  }

  function toggleAll() {
    onChange(allSelected ? [] : [...options]);
    // if deselecting all, revert to all (keep filter valid)
    if (allSelected) onChange([...options]);
  }

  return (
    <div className="ms-wrap" ref={ref}>
      <button
        type="button"
        className={`ms-trigger${open ? " ms-open" : ""}`}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="ms-label">{label}</span>
        <span className="ms-arrow">›</span>
      </button>

      {open && (
        <div className="ms-dropdown">
          <label className="ms-item ms-all">
            <input type="checkbox" checked={allSelected} onChange={toggleAll} />
            <span>{allLabel}</span>
          </label>
          <div className="ms-divider" />
          {options.map((opt) => (
            <label
              key={opt}
              className={`ms-item${value.includes(opt) ? " ms-checked" : ""}`}
            >
              <input
                type="checkbox"
                checked={value.includes(opt)}
                onChange={() => toggle(opt)}
              />
              <span>{opt}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
