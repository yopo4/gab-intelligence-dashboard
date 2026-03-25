export default function Callout({ text, kind = "sage", icon }) {
  return (
    <div className={`callout callout-${kind}`}>
      {icon && <span className="callout-icon">{icon}</span>}
      <span dangerouslySetInnerHTML={{ __html: text }} />
    </div>
  );
}
