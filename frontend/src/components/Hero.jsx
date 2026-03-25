export default function Hero({ eyebrow, titleMain, titleEm, desc, tags }) {
  return (
    <div className="page-hero">
      <div className="hero-eyebrow">{eyebrow}</div>
      <h1 className="hero-title">{titleMain} {titleEm}</h1>
      <p className="hero-desc">{desc}</p>
      <div className="tag-row">
        {tags.map((t, i) => (
          <span key={i} className={`tag tag-${t.kind || 'ghost'}`}>{t.label}</span>
        ))}
      </div>
      <div className="hero-divider" />
    </div>
  )
}
