// the site's greys for CyberChef's dark theme: same layout and blue accents, the site's near-black surfaces
const vars = {
  "--primary-background-colour": "#18181b", "--secondary-background-colour": "#131316",
  "--primary-border-colour": "#2c2c33", "--secondary-border-colour": "#26262c",
  "--title-background-colour": "#1f1f24", "--banner-bg-colour": "#131316",
  "--op-list-operation-bg-colour": "#1d1d22", "--op-list-operation-border-colour": "#2c2c33",
  "--rec-list-operation-bg-colour": "#18181b", "--rec-list-operation-border-colour": "#2c2c33",
  "--arg-background": "#232329", "--arg-border-colour": "#2c2c33",
  "--fc-operation-bg-colour": "#1d1d22", "--fc-operation-border-colour": "#2c2c33",
  "--primary-font-colour": "#d6d6dc", "--subtext-font-colour": "#8b8b96",
};
document.querySelector(":root").className = "dark";
for (const [k, v] of Object.entries(vars)) document.documentElement.style.setProperty(k, v);
