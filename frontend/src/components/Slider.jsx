export default function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
}) {
  const display = format ? format(value) : value;
  return (
    <div className="slider-wrap">
      <div className="slider-label">
        <span>{label}</span>
        <span className="slider-val">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
