import type { Metadata, Viewport } from 'next';
import Providers from './providers';
import SiteShell from '@/components/SiteShell';
import './globals.css';

export const metadata: Metadata = {
  title: 'Jaladrishti — Karnataka Reservoir Intelligence',
  description: 'Explore Karnataka reservoir observations and water-level forecasts in an immersive hydrological observatory.',
};
export const viewport: Viewport = { themeColor: '#000000', colorScheme: 'dark' };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><Providers><SiteShell>{children}</SiteShell></Providers></body></html>;
}
