export default function KpiCard({ icon, label, value, sub, color = "amber" }) {
  return (
    <div className={`kpi-card kpi-${color}`}>
      {icon && <div className="kpi-icon">{icon}</div>}
      <div className="kpi-label">{label}</div>
      <div className="kpi-val">{value}</div>
      {sub && (
        <div className="kpi-sub" dangerouslySetInnerHTML={{ __html: sub }} />
      )}
    </div>
  );
}
